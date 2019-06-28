from collections import namedtuple

from typing import Any, Dict

from flask import Flask, request, jsonify

from cassis import *

from flair.data import Token,Sentence

from flair.models import SequenceTagger

import argparse

# Types

JsonDict = Dict[str, Any]

PredictionRequest = namedtuple("PredictionRequest", ["layer", "feature", "projectId", "document", "typeSystem"])
PredictionResponse = namedtuple("PredictionResponse", ["document"])
Document = namedtuple("Document", ["xmi", "documentId", "userId"])

# Constants

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
IS_PREDICTION = "inception_internal_predicted"

# Models

parser = argparse.ArgumentParser(usage="choose ner and pos models", description="help info.")

parser.add_argument("--ner", choices=['ner', 'ner-ontonotes', 'ner-fast', 'ner-ontonotes-fast'],
                    default="ner",help="choose ner model")
parser.add_argument("--pos", choices=['pos', 'pos-fast'], default="pos", help="choose pos model")

args = parser.parse_args()

tagger: SequenceTagger = SequenceTagger.load(args.ner)
tagger_pos: SequenceTagger = SequenceTagger.load(args.pos)

# Routes

app = Flask(__name__)


@app.route("/ner/predict", methods=["POST"])
def route_predict_ner():
    json_data = request.get_json()

    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_ner(prediction_request)

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/ner/train", methods=["POST"])
def route_train_ner():
    # Return empty response
    return ('', 204)


@app.route("/pos/predict", methods=["POST"])
def route_predict_pos():
    json_data = request.get_json()

    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_pos(prediction_request)

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/pos/train", methods=["POST"])
def route_train_pos():
    # Return empty response
    return ('', 204)


def parse_prediction_request(json_object: JsonDict) -> PredictionRequest:
    metadata = json_object["metadata"]
    document = json_object["document"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    projectId = metadata["projectId"]

    xmi = document["xmi"]
    documentId = document["documentId"]
    userId = document["userId"]
    typesystem = json_object["typeSystem"]

    return PredictionRequest(layer, feature, projectId, Document(xmi, documentId, userId), typesystem)


# NLP

def predict_ner(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    # Extract the tokens from the CAS and create a flair doc from it
    tokens_cas = list(cas.select(TOKEN_TYPE))
    sentences = list(cas.select(SENTENCE_TYPE))
    text = []
    idx = 0
    for sentence in sentences:
        tokens = [Token(cas.get_covered_text(t)) for t in list(cas.select_covered(TOKEN_TYPE,sentence))]
        for token in tokens:
            token.idx = idx
            idx+=1
        s = Sentence()
        s.tokens = tokens
        text.append(s)
    tagger.predict(text)

    # Find the named entities
    for sen in text:
        for ent in sen.get_spans('ner'):
            start_idx = ent.tokens[0].idx

            end_idx = start_idx + len(ent.tokens) -1
            fields = {'begin': tokens_cas[start_idx].begin,
                      'end': tokens_cas[end_idx].end,
                      IS_PREDICTION: True,
                      prediction_request.feature: ent.tag}
            annotation = AnnotationType(**fields)
            cas.add_annotation(annotation)

    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


def predict_pos(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    # Extract the tokens from the CAS and create a spacy doc from it
    tokens_cas = list(cas.select(TOKEN_TYPE))
    sentences = list(cas.select(SENTENCE_TYPE))
    text = []
    idx = 0
    for sentence in sentences:
        tokens = [Token(cas.get_covered_text(t)) for t in list(cas.select_covered(TOKEN_TYPE,sentence))]
        for token in tokens:
            token.idx = idx
            idx+=1
        s = Sentence()
        s.tokens = tokens
        text.append(s)

    # Do the tagging
    tagger_pos.predict(text)

    # For every token, extract the POS tag and create an annotation in the CAS
    for sen in text:
        for token in sen:
            for t in token.tags.values():
                fields = {'begin': tokens_cas[token.idx].begin,
                          'end': tokens_cas[token.idx].end,
                          IS_PREDICTION: True,
                          prediction_request.feature: t.value}
                annotation = AnnotationType(**fields)
                cas.add_annotation(annotation)

    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    """
    # For debugging purposes, load a json file containing the request and process it.
    import json
    with open("predict.json", "rb") as f:
        predict_json = json.load(f)

    request = parse_prediction_request(predict_json)
    predict_pos(request)
    """

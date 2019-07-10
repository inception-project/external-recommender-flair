from collections import namedtuple

from typing import Any, Dict

from flask import Flask, request, jsonify

from cassis import *

from flair.data import Token, Sentence

from flair.models import SequenceTagger

import argparse

import logging

import gunicorn.app.base

from gunicorn.six import iteritems

import time

from multiprocessing import Lock

# Types

JsonDict = Dict[str, Any]

PredictionRequest = namedtuple("PredictionRequest", ["layer", "feature", "projectId", "document", "typeSystem"])
PredictionResponse = namedtuple("PredictionResponse", ["document"])
Document = namedtuple("Document", ["xmi", "documentId", "userId"])

# Constants

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
IS_PREDICTION = "inception_internal_predicted"

# Locks
lock_ner_train = Lock()
lock_pos_train = Lock()

# Models

parser = argparse.ArgumentParser(usage="choose ner and pos models", description="help info.")

parser.add_argument("--ner", choices=['ner', 'ner-ontonotes', 'ner-fast', 'ner-ontonotes-fast'],
                    default="ner", help="choose ner model")
parser.add_argument("--pos", choices=['pos', 'pos-fast'], default="pos", help="choose pos model")

parser.add_argument("--workers", default=1)

parser.add_argument("--threads", default=1)

args = parser.parse_args()

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

tagger: SequenceTagger = SequenceTagger.load(args.ner)
tagger_pos: SequenceTagger = SequenceTagger.load(args.pos)

# Routes

app = Flask(__name__)


@app.route("/ner/predict", methods=["POST"])
def route_predict_ner():
    json_data = request.get_json()

    logging.warning('ner prediction starts')
    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_ner(prediction_request)
    logging.warning('ner prediction ends')

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/ner/train", methods=["POST"])
def route_train_ner():
    if (lock_ner_train.acquire(block=False)):
        # Return empty response
        try:
            logging.warning('ner train starts')
            time.sleep(15)
            logging.warning('ner train ends')
        finally:
            lock_ner_train.release()
            return ('', 204)
    else:
        return ('', 429)


@app.route("/pos/predict", methods=["POST"])
def route_predict_pos():
    json_data = request.get_json()

    logging.warning('pos prediction starts')
    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_pos(prediction_request)
    logging.warning('pos prediction ends')

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/pos/train", methods=["POST"])
def route_train_pos():
    if (lock_pos_train.acquire(block=False)):
        # Return empty response
        try:
            logging.warning('pos train starts')
            time.sleep(15)
            logging.warning('pos train ends')
        finally:
            lock_pos_train.release()
            return ('', 204)
    else:
        return ('', 429)


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
        tokens = [Token(cas.get_covered_text(t)) for t in list(cas.select_covered(TOKEN_TYPE, sentence))]
        for token in tokens:
            token.idx = idx
            idx += 1
        s = Sentence()
        s.tokens = tokens
        text.append(s)
    tagger.predict(text)

    # Find the named entities
    for sen in text:
        for ent in sen.get_spans('ner'):
            start_idx = ent.tokens[0].idx

            end_idx = start_idx + len(ent.tokens) - 1
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
        tokens = [Token(cas.get_covered_text(t)) for t in list(cas.select_covered(TOKEN_TYPE, sentence))]
        for token in tokens:
            token.idx = idx
            idx += 1
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


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    ##app.run(debug=True, host='0.0.0.0')
    # app.run(host='0.0.0.0')

    options = {
        'bind': '%s:%s' % ('0.0.0.0', '5000'),
        'workers': args.workers,
        'threads': args.threads,
    }
    StandaloneApplication(app, options).run()
    """

    # For debugging purposes, load a json file containing the request and process it.
    import json
    with open("predict.json", "rb") as f:
        predict_json = json.load(f)

    request = parse_prediction_request(predict_json)
    predict_pos(request)
    """

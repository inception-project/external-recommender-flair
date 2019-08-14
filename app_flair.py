from collections import namedtuple

from typing import Any, Dict

from flask import Flask, request, jsonify

from cassis import *

from flair.data import Token, Sentence

from flair.models import SequenceTagger

from flair.data import Corpus

from flair.trainers import ModelTrainer

from multiprocessing import Lock

from sklearn.model_selection import train_test_split

import argparse

import os

# Types

JsonDict = Dict[str, Any]

PredictionRequest = namedtuple("PredictionRequest", ["layer", "feature", "projectId", "document", "typeSystem"])
PredictionResponse = namedtuple("PredictionResponse", ["document"])
Document = namedtuple("Document", ["xmi", "documentId", "userId"])

# Constants

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
NER_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
POS_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
IS_PREDICTION = "inception_internal_predicted"

# Locks

lock_ner_train = Lock()
lock_pos_train = Lock()

# Models

global tagger_ner

global tagger_pos

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
    if (lock_ner_train.acquire(block=False)):
        try:
            global tagger_ner
            json_data = request.get_json()
            train, dev = parse_ner_train_request(json_data)
            corpus = Corpus( train=train, dev=dev, test="")
            trainer: ModelTrainer = ModelTrainer(tagger_ner,corpus)
            trainer.train('model_ner',learning_rate = 0.1, mini_batch_size= 16, max_epochs = 10,
                          save_final_model=False)
            tagger_ner = SequenceTagger.load('model_ner/best-model.pt')
        finally:
            lock_ner_train.release()
            return ('', 204)
    else:
        return ('', 429)


@app.route("/pos/predict", methods=["POST"])
def route_predict_pos():
    json_data = request.get_json()
    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_pos(prediction_request)

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/pos/train", methods=["POST"])
def route_train_pos():
    json_data = request.get_json()
    if lock_pos_train.acquire(block = False):
        try:
            global tagger_pos
            train, dev = parse_pos_train_request(json_data)
            corpus = Corpus(train=train, dev=dev, test="")
            trainer: ModelTrainer = ModelTrainer(tagger_pos, corpus)
            trainer.train('model_pos', learning_rate=0.1, mini_batch_size=16, max_epochs= 10,
                          save_final_model=False)
            tagger_pos = SequenceTagger.load('model_pos/best-model.pt')
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


def parse_ner_train_request(json_object: JsonDict):
    documents = json_object['documents']
    list_sentences = []
    for document in documents:
        typesystem = load_typesystem(json_object["typeSystem"])
        cas = load_cas_from_xmi(document['xmi'], typesystem=typesystem)
        tagset = [b.decode('utf-8') for b in list(tagger_ner.tag_dictionary.idx2item)]
        sentence_list = cas.select(SENTENCE_TYPE)
        for sentence in sentence_list:
            i = 0
            j = 0
            token_list = list(cas.select_covered(TOKEN_TYPE,sentence))
            ner_list = list(cas.select_covered(NER_TYPE, sentence))
            token_list_len = len(token_list)
            ner_list_len = len(ner_list)
            tokens = []
            flag = False
            while i < token_list_len and j < ner_list_len:
                if "S-" + ner_list[j].value not in tagset:
                    flag = True
                    break
                token = Token(cas.get_covered_text(token_list[i]))
                if token_list[i].begin == ner_list[j].begin and token_list[i].end == ner_list[j].end:
                    token.add_tag('ner',"S-" + ner_list[j].value)
                    i += 1
                    j += 1
                elif token_list[i].begin == ner_list[j].begin and token_list[i].end < ner_list[j].end:
                    token.add_tag('ner',"B-" + ner_list[j].value)
                    i += 1
                elif token_list[i].begin > ner_list[j].begin and token_list[i].end < ner_list[j].end:
                    token.add_tag('ner',"I-" + ner_list[j].value)
                    i += 1
                elif token_list[i].begin > ner_list[j].begin and token_list[i].end == ner_list[j].end:
                    token.add_tag('ner',"E-" + ner_list[j].value)
                    i += 1
                    j += 1
                else:
                    token.add_tag('ner','O')
                    i += 1
                tokens.append(token)
            if flag or ner_list_len == 0:
                continue
            while i < token_list_len:
                token = Token(cas.get_covered_text(token_list[i]))
                token.add_tag('ner', 'O')
                i += 1
                tokens.append(token)
            s = Sentence()
            s.tokens = tokens
            list_sentences.append(s)
    train, dev, = train_test_split( list_sentences, train_size=0.8)
    return train, dev


def parse_pos_train_request(json_object: JsonDict):
    documents = json_object['documents']
    tagset = [b.decode('utf-8') for b in list(tagger_pos.tag_dictionary.idx2item)]
    list_sentences = []
    for document in documents:
        typesystem = load_typesystem(json_object["typeSystem"])
        cas = load_cas_from_xmi(document['xmi'], typesystem=typesystem)
        sentence_list = cas.select(SENTENCE_TYPE)
        for sentence in sentence_list:
            tokens = []
            pos_list = cas.select_covered(POS_TYPE,sentence)
            token_list = cas.select_covered(TOKEN_TYPE,sentence)
            if len(pos_list) != len(token_list):
                continue
            flag = False
            for pos in pos_list:
                if pos.PosValue not in tagset:
                    flag = True
                    break
                token = Token(cas.get_covered_text(pos))
                token.add_tag('pos',pos.PosValue)
                tokens.append(token)
            if flag:
                continue
            s = Sentence()
            s.tokens = tokens
            list_sentences.append(s)
    train, dev = train_test_split( list_sentences, train_size=0.8)
    return train, dev


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
    tagger_ner.predict(text)

    # Find the named entities
    for sen in text:
        for ent in sen.get_spans('ner'):
            start_idx = ent.tokens[0].idx

            end_idx = start_idx + len(ent.tokens) - 1
            fields = {'begin': tokens_cas[start_idx].begin,
                      'end': tokens_cas[end_idx].end,
                      IS_PREDICTION: True,
                      prediction_request.feature+"_score": ent.score,
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
                          prediction_request.feature+"_score": t.score,
                          prediction_request.feature: t.value}
                annotation = AnnotationType(**fields)
                cas.add_annotation(annotation)

    xmi = cas.to_xmi()
    return PredictionResponse(xmi)

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(usage="choose ner and pos models", description="help info.")

    parser.add_argument("--ner", choices=['ner', 'ner-ontonotes', 'ner-fast', 'ner-ontonotes-fast'],
                        default="ner",help="choose ner model")
    parser.add_argument("--pos", choices=['pos', 'pos-fast'], default="pos", help="choose pos model")

    args = parser.parse_args()

    tagger_ner = SequenceTagger.load(args.ner)

    tagger_pos= SequenceTagger.load(args.pos)

    app.run(debug=True, host='0.0.0.0')
    """

    # For debugging purposes, load a json file containing the request and process it.
    import json
    with open("predict.json", "rb") as f:
        predict_json = json.load(f)

    request = parse_prediction_request(predict_json)
    predict_pos(request)
    """
else:
    tagger_ner = SequenceTagger.load(os.getenv('ner_model'))

    tagger_pos = SequenceTagger.load(os.getenv('pos_model'))

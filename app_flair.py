from collections import namedtuple

from typing import Any, Dict, List

from flask import Flask, request, jsonify

from cassis import *

from flair.data import Token, Sentence

from flair.models import SequenceTagger

from flair.data import Corpus

from flair.trainers import ModelTrainer

from multiprocessing import Lock

import threading

from sklearn.model_selection import train_test_split

import argparse

import copy

import os

from http import HTTPStatus

# Types

JsonDict = Dict[str, Any]

PredictionRequest = namedtuple(
    "PredictionRequest", ["layer", "feature", "projectId", "document", "typeSystem"]
)
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

# Routes

app = Flask(__name__)


@app.route("/ner/predict", methods=["POST"])
def route_predict_ner():
    # Deal with the ner prediction request
    json_data = request.get_json()
    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_ner(prediction_request)

    result = jsonify(document=prediction_response.document)
    return result


@app.route("/ner/train", methods=["POST"])
def route_train_ner():
    # Deal with the ner training request
    # Use the lock to make sure that only one training is in the process for each time
    # The training is in a background thread
    if lock_ner_train.acquire(block=False):
        json_data = request.get_json()
        train, dev = parse_ner_train_request(json_data)
        if train is not None and dev is not None:
            t1 = threading.Thread(target=flair_train_ner, args=(train, dev))
            t1.start()
        else:
            lock_ner_train.release()
        return HTTPStatus.NO_CONTENT.description, HTTPStatus.NO_CONTENT.value
    else:
        return HTTPStatus.TOO_MANY_REQUESTS.description, HTTPStatus.TOO_MANY_REQUESTS.value


@app.route("/pos/predict", methods=["POST"])
def route_predict_pos():
    # Deal with the pos prediction request
    json_data = request.get_json()
    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_pos(prediction_request)

    result = jsonify(document=prediction_response.document)
    return result


@app.route("/pos/train", methods=["POST"])
def route_train_pos():
    # Deal with the pos training request
    # Use the lock to make sure that only one training is in the process for each time
    # The training is in a background thread
    if lock_pos_train.acquire(block=False):
        json_data = request.get_json()
        train, dev = parse_pos_train_request(json_data)
        if train is not None and dev is not None:
            t2 = threading.Thread(target=flair_train_pos, args=(train, dev))
            t2.start()
        else:
            lock_pos_train.release()
        return HTTPStatus.NO_CONTENT.description, HTTPStatus.NO_CONTENT.value
    else:
        return HTTPStatus.TOO_MANY_REQUESTS.description, HTTPStatus.TOO_MANY_REQUESTS.value


class Model:
    def __init__(self, ner_model, pos_model):
        self.ner_model = ner_model
        self.pos_model = pos_model

    def get_ner_model(self):
        return self.ner_model

    def get_pos_model(self):
        return self.pos_model


def parse_prediction_request(json_object: JsonDict) -> PredictionRequest:
    # Parse the request into a prediction request
    metadata = json_object["metadata"]
    document = json_object["document"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    projectId = metadata["projectId"]

    xmi = document["xmi"]
    documentId = document["documentId"]
    userId = document["userId"]
    typesystem = json_object["typeSystem"]

    return PredictionRequest(
        layer, feature, projectId, Document(xmi, documentId, userId), typesystem
    )


def parse_ner_train_request(json_object: JsonDict) -> (List[Sentence], List[Sentence]):
    # Extract the ner-tagged sentences from each documents of the training request
    # Split the sentences into training set and development set
    documents = json_object["documents"]
    list_sentences = []
    typesystem = load_typesystem(json_object["typeSystem"])
    for document in documents:
        cas = load_cas_from_xmi(document["xmi"], typesystem=typesystem)
        list_from_document = flair_train_ner_dataset(cas)
        list_sentences.extend(list_from_document)
    if len(list_sentences) > 1:
        train, dev, = train_test_split(list_sentences, train_size=0.8)
        return train, dev
    else:
        return None, None


def flair_train_ner_dataset(cas: Cas) -> List[Sentence]:
    # In flair the instance of Sentence is used to train the model
    # Each document is transformed into a list of instances of Sentence with the BIO encoding of NER tags
    # If a sentence in the document has the wrong NER value which is not in the tagset, the sentence will be discarded
    tagset = [b.decode("utf-8") for b in model.get_ner_model().tag_dictionary.idx2item]
    sentence_list = cas.select(SENTENCE_TYPE)
    list_sentences = []
    for sentence in sentence_list:
        tokens = bio_encoding(cas, sentence, tagset, Token)
        if tokens is None:
            continue
        s = Sentence()
        s.tokens = tokens
        list_sentences.append(s)

    return list_sentences


def bio_encoding(cas, sentence, tagset, t):
    i = 0
    j = 0
    token_list = list(cas.select_covered(TOKEN_TYPE, sentence))
    ner_list = list(cas.select_covered(NER_TYPE, sentence))
    token_list_len = len(token_list)
    ner_list_len = len(ner_list)
    tokens = []
    incomplete_sentence = False

    # In the BIO encoding, "B-tag" is the begin of entity, "I-tag" is the continuation of entity
    # and "O" is no entity.
    if ner_list_len == 0:
        incomplete_sentence = True
    else:
        while i < token_list_len and j < ner_list_len:
            if ner_list[j].value is None or "B-" + ner_list[j].value not in tagset:
                incomplete_sentence = True
                break
            token = t(cas.get_covered_text(token_list[i]))
            if token_list[i].begin == ner_list[j].begin:
                token.add_tag("ner", "B-" + ner_list[j].value)
                if token_list[i].end == ner_list[j].end:
                    j += 1
                i += 1
            elif (
                    token_list[i].begin > ner_list[j].begin
                    and token_list[i].end <= ner_list[j].end
            ):
                token.add_tag("ner", "I-" + ner_list[j].value)
                if token_list[i].end == ner_list[j].end:
                    j += 1
                i += 1
            else:
                token.add_tag("ner", "O")
                i += 1
            tokens.append(token)
    if incomplete_sentence:
        return None
    while i < token_list_len:
        token = Token(cas.get_covered_text(token_list[i]))
        token.add_tag("ner", "O")
        i += 1
        tokens.append(token)
    return tokens


def flair_train_ner(train: List[Sentence], dev: List[Sentence]):
    # Use a copy of the model for training because during the training the prediction should still work
    # There is no need to set the test set because the training is on train set and evaluation is on development set
    corpus = Corpus(train=train, dev=dev, test="")

    tagger_ner_for_train = copy.deepcopy(model.get_ner_model())
    trainer: ModelTrainer = ModelTrainer(tagger_ner_for_train, corpus)

    # As for the hyper-parameters of the model
    # base_path: Main path to which all output during training is logged and models are saved
    # learning_rate: Initial learning rate
    # mini_batch_size: Size of mini-batches during training.
    # max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
    # anneal_factor: The factor by which the learning rate is annealed.
    # new learning rate = learning rate * anneal_factor
    # Default minimal learning rate is 0.0001. If the learning rate falls below this threshold, training terminates.
    # patience: Patience is the number of epochs with no improvement the Trainer waits
    # until annealing the learning rate
    trainer.train(
        "model_ner",
        learning_rate=0.1,
        mini_batch_size=16,
        max_epochs=150,
        anneal_factor=0.1,
        patience=1,
        save_final_model=False,
    )

    # After the training, the tagger model will be updated into the best model from the training.
    model.ner_model = SequenceTagger.load("model_ner/best-model.pt")
    # Release the lock and be able to deal with new training request
    lock_ner_train.release()


def parse_pos_train_request(json_object: JsonDict) -> (List[Sentence], List[Sentence]):
    # Extract the pos-tagged sentences from each documents of the training request
    # Split the sentences into training set and development set
    documents = json_object["documents"]
    list_sentences = []
    typesystem = load_typesystem(json_object["typeSystem"])
    for document in documents:
        cas = load_cas_from_xmi(document["xmi"], typesystem=typesystem)
        list_from_document = flair_train_pos_dataset(cas)
        list_sentences.extend(list_from_document)
    if len(list_sentences) > 1:
        train, dev = train_test_split(list_sentences, train_size=0.8)
        return train, dev
    else:
        return None, None


def flair_train_pos_dataset(cas: Cas):
    # In flair the instance of Sentence is used to train the model
    # Each document is transformed into a list of instances of Sentence with the POS tags
    # If a sentence in the document has the wrong POS value which is not in the tagset, the sentence will be discarded
    tagset = [b.decode("utf-8") for b in model.pos_model.tag_dictionary.idx2item]
    sentence_list = cas.select(SENTENCE_TYPE)
    list_sentences = []
    for sentence in sentence_list:
        tokens = []
        pos_list = list(cas.select_covered(POS_TYPE, sentence))
        token_list = list(cas.select_covered(TOKEN_TYPE, sentence))
        if len(pos_list) != len(token_list):
            continue
        incomplete_sentence = False
        for pos in pos_list:
            if pos.PosValue not in tagset:
                incomplete_sentence = True
                break
            token = Token(cas.get_covered_text(pos))
            token.add_tag("pos", pos.PosValue)
            tokens.append(token)
        if incomplete_sentence:
            continue
        s = Sentence()
        s.tokens = tokens
        list_sentences.append(s)
    return list_sentences


def flair_train_pos(train: List[Sentence], dev: List[Sentence]):
    # Use a copy of the model for training because during the training the prediction should still work
    # There is no need to set the test set because the training is on train set and evaluation is on development set
    corpus = Corpus(train=train, dev=dev, test="")
    tagger_pos_for_train = copy.deepcopy(model.get_pos_model())
    trainer: ModelTrainer = ModelTrainer(tagger_pos_for_train, corpus)

    # As for the hyper-parameters of the model
    # base_path: Main path to which all output during training is logged and models are saved
    # learning_rate: Initial learning rate
    # mini_batch_size: Size of mini-batches during training.
    # max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
    # anneal_factor: The factor by which the learning rate is annealed.
    # new learning rate = learning rate * anneal_factor
    # Default minimal learning rate is 0.0001. If the learning rate falls below this threshold, training terminates.
    # patience: Patience is the number of epochs with no improvement the Trainer waits
    # until annealing the learning rate
    trainer.train(
        base_path="model_pos",
        learning_rate=0.1,
        mini_batch_size=16,
        max_epochs=150,
        anneal_factor=0.1,
        patience=1,
        save_final_model=False,
    )

    # After the training, the tagger model will be updated into the best model from the training.
    model.pos_model = SequenceTagger.load("model_pos/best-model.pt")
    # Release the lock and be able to deal with new training request
    lock_pos_train.release()


# NLP


def predict_ner(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    cas = flair_predict_ner(cas, AnnotationType, prediction_request)
    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


def flair_predict_ner(cas, annotationtype, prediction_request):
    # Extract the tokens from the CAS and create a flair doc from it
    tokens_cas = list(cas.select(TOKEN_TYPE))
    sentences = cas.select(SENTENCE_TYPE)
    text = []
    idx = 0
    for sentence in sentences:
        tokens = [
            Token(cas.get_covered_text(t))
            for t in list(cas.select_covered(TOKEN_TYPE, sentence))
        ]
        for token in tokens:
            token.idx = idx
            idx += 1
        s = Sentence()
        s.tokens = tokens
        text.append(s)
    model.get_ner_model().predict(text)

    # Find the named entities
    for sen in text:
        for ent in sen.get_spans("ner"):
            start_idx = ent.tokens[0].idx

            end_idx = start_idx + len(ent.tokens) - 1
            fields = {
                "begin": tokens_cas[start_idx].begin,
                "end": tokens_cas[end_idx].end,
                IS_PREDICTION: True,
                prediction_request.feature + "_score": ent.score,
                prediction_request.feature: ent.tag,
            }
            annotation = annotationtype(**fields)
            cas.add_annotation(annotation)
    return cas


def predict_pos(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    cas = flair_predict_pos(cas, AnnotationType, prediction_request)
    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


def flair_predict_pos(cas, annotationtype, prediction_request):
    # Extract the tokens from the CAS and create a flair doc from it
    tokens_cas = list(cas.select(TOKEN_TYPE))
    sentences = cas.select(SENTENCE_TYPE)
    text = []
    idx = 0
    for sentence in sentences:
        tokens = [
            Token(cas.get_covered_text(t))
            for t in list(cas.select_covered(TOKEN_TYPE, sentence))
        ]
        for token in tokens:
            token.idx = idx
            idx += 1
        s = Sentence()
        s.tokens = tokens
        text.append(s)

    # Do the tagging
    model.get_pos_model().predict(text)

    # For every token, extract the POS tag and create an annotation in the CAS
    for sen in text:
        for token in sen:
            for t in token.tags.values():
                fields = {
                    "begin": tokens_cas[token.idx].begin,
                    "end": tokens_cas[token.idx].end,
                    IS_PREDICTION: True,
                    prediction_request.feature + "_score": t.score,
                    prediction_request.feature: t.value,
                }
                annotation = annotationtype(**fields)
                cas.add_annotation(annotation)
    return cas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="choose ner and pos models", description="help info."
    )

    parser.add_argument(
        "--ner",
        choices=["ner", "ner-ontonotes", "ner-fast", "ner-ontonotes-fast"],
        default="ner",
        help="choose ner model",
    )
    parser.add_argument(
        "--pos", choices=["pos", "pos-fast"], default="pos", help="choose pos model"
    )

    args = parser.parse_args()

    model = Model(SequenceTagger.load(args.ner),SequenceTagger.load(args.pos))

    app.run(debug=True, host="0.0.0.0")

elif "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    # start the application with gunicorn
    model = Model(SequenceTagger.load(os.getenv("ner_model")),SequenceTagger.load(os.getenv("pos_model")))

else:
    # used in the test case
    model = Model(SequenceTagger.load('ner'), SequenceTagger.load('pos'))
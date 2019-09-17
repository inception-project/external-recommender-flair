from pathlib import Path
from flair.datasets import ColumnDataset
import json
from cassis import *
import unittest
from app_flair import flair_train_ner_dataset


class TestBIOEncoding(unittest.TestCase):
    # Test on the BIO encoding function
    def test_sentences(self):
        column_name = {0: "text", 1: "ner"}
        corpus = ColumnDataset(Path("tests/test_n.txt"), column_name)
        list_s = []
        for i in corpus:
            if len(i.get_spans("ner")) > 0:
                list_s.append(i)
            else:
                continue

        with open('tests/test.json', 'r') as json_file:
            json_object = json.load(json_file)
        document = json_object["documents"][0]
        typesystem = load_typesystem(json_object["typeSystem"])
        cas = load_cas_from_xmi(document['xmi'], typesystem=typesystem)
        list_sentences = flair_train_ner_dataset(cas)
        for sentence, t_sentence in zip(list_s, list_sentences):
            self.assertEqual(
                sentence.to_tagged_string("ner"), t_sentence.to_tagged_string("ner")
            )
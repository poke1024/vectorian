# This is a sample Python script.

import vectorian.core as core
import vectorian.utils as utils
import vectorian

from vectorian.metrics import CosineMetric, TokenSimilarityMetric, AlignmentSentenceMetric
from vectorian.importers import NovelImporter
from vectorian.embeddings import FastText
from vectorian.session import Session
from vectorian.corpus import Corpus
from vectorian.render import LocationFormatter

import spacy
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    embedding = FastText("en")

    nlp = spacy.load("en_core_web_sm")

    # use case 1.
    if False:
        im = NovelImporter(nlp)
        doc = im("/Users/arbeit/A Child's Dream of a Star.txt")

        session = Session(
            [doc],
            [embedding])

    # use case 2.
    im = NovelImporter(nlp)

    corpus = Corpus()
    doc = im("/Users/arbeit/Wise Children.txt")
    corpus.add(doc)
    corpus.save("/Users/arbeit/Desktop/my-corpus")

    token_mappings = {
        "tokenizer": [],
        "tagger": []
    }

    token_mappings["tokenizer"].append(utils.lowercase())
    token_mappings["tokenizer"].append(utils.erase("W"))
    token_mappings["tokenizer"].append(utils.alpha())

    session = Session(
        corpus,
        [embedding],
        token_mappings)

    #doc.save("/Users/arbeit/temp.json")
    #cdoc = doc.to_core(0, vocab)
    #print(cdoc)

    #session.add_document(doc)

    formatter = LocationFormatter()

    metric = AlignmentSentenceMetric(
        TokenSimilarityMetric(
            embedding, CosineMetric()))

    #p = Partition("sentence")
    index = session.partition("token", 25, 1).index(metric, nlp)
    matches = index.find("write female", n=3)

    #index = session.index_for_metric("auto", nlp=nlp)
    #matches = index.find("company")
    with open("/Users/arbeit/Desktop/temp.json", "w") as f:
        f.write(json.dumps(matches.to_json(10, formatter), indent=4))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

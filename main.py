# This is a sample Python script.

import vectorian.core as core
import vectorian

from vectorian.importers import NovelImporter
from vectorian.embeddings import FastText
from vectorian.session import Session
from vectorian.corpus import Corpus

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
    doc = im("/Users/arbeit/A Child's Dream of a Star.txt")
    corpus.add(doc)
    corpus.save("/Users/arbeit/Desktop/my-corpus")

    session = Session(
        corpus,
        [embedding])

    #doc.save("/Users/arbeit/temp.json")
    #cdoc = doc.to_core(0, vocab)
    #print(cdoc)

    #session.add_document(doc)

    query = nlp("company")
    matches = session.find(query)
    with open("/Users/arbeit/Desktop/temp.json", "w") as f:
        f.write(json.dumps(matches.to_json(), indent=4))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

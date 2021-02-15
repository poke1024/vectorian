# This is a sample Python script.

import vectorian.core as core
import vectorian

import spacy
from vectorian.importers import NovelImporter
from vectorian.embeddings import FastText

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(core)

    vocab = core.Vocabulary()
    embedding = FastText("en")

    nlp = spacy.load("en_core_web_sm")
    im = NovelImporter(nlp)
    doc = im("/Users/arbeit/A Child's Dream of a Star.txt")
    #doc.save("/Users/arbeit/temp.json")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

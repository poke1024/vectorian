from pathlib import Path

import os
import sys
import json

script_dir = Path(os.path.realpath(__file__)).parent
sys.path.append(str(script_dir))

import spacy
nlp = spacy.load("en_core_web_sm")

import vectorian
vectorian.compile(for_debugging=True)

import vectorian.utils as utils
from vectorian.metrics import CosineMetric, TokenSimilarityMetric, AlignmentSentenceMetric
from vectorian.importers import NovelImporter
from vectorian.embeddings import PretrainedFastText
from vectorian.session import Session
from vectorian.corpus import Corpus
from vectorian.render.location import LocationFormatter

fasttext = PretrainedFastText("en")

# use case 1.
if False:
    im = NovelImporter(nlp)
    doc = im("/Users/arbeit/A Child's Dream of a Star.txt")

    session = Session(
        [doc],
        [fasttext])

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
    [fasttext],
    token_mappings)

# doc.save("/Users/arbeit/temp.json")
# cdoc = doc.to_core(0, vocab)
# print(cdoc)

# session.add_document(doc)

formatter = LocationFormatter()

metric = AlignmentSentenceMetric(
    TokenSimilarityMetric(
        fasttext, CosineMetric()))

# p = Partition("sentence")
index = session.partition("token", 25, 1).index(metric, nlp)
matches = index.find("write female", n=3)

# index = session.index_for_metric("auto", nlp=nlp)
# matches = index.find("company")
with open("/Users/arbeit/Desktop/temp.json", "w") as f:
    f.write(json.dumps(matches.to_json(), indent=4))




from vectorian.metrics import CosineMetric, TokenSimilarityMetric, AlignmentSentenceMetric
from vectorian.alignment import WordRotatorsDistance
import tabulate


debug_data = []


def debug_hook(hook, data):
    # numpy arrays here might be mapped, copy them for later processing.
    if False:
        debug_data.append({
            's': data['s'],
            't': data['t'],
            'D': data['D'].copy(),
            'G': data['G'].copy()})


wrd_metric = AlignmentSentenceMetric(
    token_metric=TokenSimilarityMetric(fasttext, CosineMetric()),
    alignment=WordRotatorsDistance())
index = session.partition("sentence").index(wrd_metric, nlp)

r = index.find("day", n=5, min_score=0.1, options={'debug': debug_hook})

with open("/Users/arbeit/Desktop/debug.txt", "w") as f:
    def write_table(name, data):
        f.write("\n")
        f.write(name + ":\n")
        table = [[""] + x["t"]]
        for i in range(len(x["s"])):
            row = [x["s"][i]]
            for j in range(len(x["t"])):
                row.append("%.4f" % data[i, j])
            table.append(row)
        f.write(tabulate.tabulate(table) + "\n")

    for x in debug_data:
        f.write("s: " + " ".join(x["s"]) + " (" + str(len(x["s"])) + ")\n")
        f.write("t: " + " ".join(x["t"]) + " (" + str(len(x["t"])) + ")\n")
        write_table("D", x["D"])
        write_table("G", x["G"])
        f.write("\n\n")


import spacy

from pathlib import Path

import os
import sys

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir.parent))

import vectorian

from vectorian.importers import NovelImporter
from vectorian.session import LabSession
from vectorian.embeddings import SpacyTransformerEmbedding
from vectorian.importers import StringImporter
from vectorian.session import LabSession
from vectorian.metrics import CosineSimilarity, TokenSimilarity
from vectorian.metrics import AlignmentSentenceSimilarity
from vectorian.alignment import WatermanSmithBeyer, ConstantGapCost

nlp = spacy.load("en_core_web_trf")

my_embedding = SpacyTransformerEmbedding(nlp)

# contextual embeddings
im = NovelImporter(nlp, embeddings=[my_embedding])

#im = NovelImporter(nlp)
doc = im("/Users/arbeit/Wise Children Mini.txt")

# note: you may also specify StackedEmbedding here.
session = LabSession(
    [doc],
    embeddings=[my_embedding],
    normalizers="default")

metric = AlignmentSentenceSimilarity(
    token_metric=TokenSimilarity(my_embedding, CosineSimilarity()),
    alignment=WatermanSmithBeyer(gap=ConstantGapCost(0), zero=0.25))

index = session.partition("sentence").index(metric, nlp)

index.find('thing', n=1)

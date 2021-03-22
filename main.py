# This is a sample Python script.

import vectorian.utils as utils

from vectorian.metrics import CosineSimilarity, TokenSimilarity, MixedTokenSimilarity, AlignmentSentenceSimilarity
from vectorian.alignment import WordMoversDistance, WordRotatorsDistance
from vectorian.importers import NovelImporter
from vectorian.embeddings import PretrainedFastText, CompressedFastTextVectors, PretrainedGloVe
from vectorian.session import Session
from vectorian.corpus import Corpus
from vectorian.render.location import LocationFormatter

import spacy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #fasttext = PretrainedFastText("en")
    fasttext = CompressedFastTextVectors("models/fasttext-en-mini.kv")
    glove = PretrainedGloVe("6B", 50)

    nlp = spacy.load("en_core_web_sm")

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

    #token_mappings["tokenizer"].append(utils.lowercase())
    token_mappings["tokenizer"].append(utils.erase("W"))
    token_mappings["tokenizer"].append(utils.alpha())

    session = Session(
        corpus,
        [fasttext, glove],
        token_mappings)

    #doc.save("/Users/arbeit/temp.json")
    #cdoc = doc.to_core(0, vocab)
    #print(cdoc)

    #session.add_document(doc)

    formatter = LocationFormatter()

    if False:
        metric = AlignmentSentenceMetric(
            TokenSimilarityMetric(
                fasttext, CosineMetric()))

        index = session.partition("token", 25, 1).index(metric, nlp)
        matches = index.find("write female", n=3)
    else:
        import numpy as np
        import json

        with open("/Users/arbeit/debug.txt", "w") as f:
            def debug(hook, args):
                if hook != 'alignment/wmd':
                    return

                for k, v in args.items():
                    f.write(f"{k}:\n")
                    if isinstance(v, np.ndarray):
                        f.write(np.array2string(v))
                    else:
                        f.write(json.dumps(v, indent=4))
                    f.write("\n")
                    f.write("\n")

                f.write("-" * 80)
                f.write("\n")
                f.write("\n")

            if True:
                metric = MixedTokenSimilarity(
                    [
                        TokenSimilarity(fasttext, CosineSimilarity()),
                        TokenSimilarity(glove, CosineSimilarity())
                    ],
                    [1, 1]
                )

                metric = TokenSimilarity(fasttext, CosineSimilarity())

                index = session.partition("sentence").index(AlignmentSentenceSimilarity(
                    token_metric=metric,
                    alignment=WordMoversDistance.wmd('kusner')), nlp=nlp)
            else:
                index = session.partition("sentence").index(AlignmentSentenceMetric(
                    token_metric=TokenSimilarityMetric(fasttext, CosineSimilarity()),
                    alignment=WordRotatorsDistance()), nlp=nlp)

            #matches = index.find("write female", n=3, debug=debug)
            #matches = index.find("the great star", n=3, debug=debug)
            matches = index.find("speak", n=3)

    #index = session.index_for_metric("auto", nlp=nlp)
    #matches = index.find("company")
    with open("/Users/arbeit/Desktop/temp.json", "w") as f:
        f.write(json.dumps(matches.to_json(10), indent=4))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

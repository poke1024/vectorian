# This is a sample Python script.

import vectorian.core as core
import vectorian

from vectorian.importers import NovelImporter
from vectorian.embeddings import FastText
from vectorian.session import Session
from vectorian.corpus import Corpus

import spacy
import traceback
import json
import roman


def get_location_desc(metadata, location):
    if location[2] > 0:  # we have an act-scene-speakers structure.
        speaker = metadata["speakers"].get(str(location[2]), "")
        if location[0] >= 0:
            act = roman.toRoman(location[0])
            scene = location[1]
            return speaker, "%s.%d, line %d" % (act, scene, location[3])
        else:
            return speaker, "line %d" % location[3]
    elif location[1] > 0:  # book, chapter and paragraphs
        if location[0] < 0:  # do we have a book?
            return "", "Chapter %d, par. %d" % (location[1], location[3])
        else:
            return "", "Book %d, Chapter %d, par. %d" % (
                location[0], location[1], location[3])
    else:
        return "", "par. %d" % location[3]


def rs_to_matches(result_set):
    matches = []
    for i, m in enumerate(result_set.best_n(5)):

        regions = []

        try:
            for r in m.regions:
                s = r.s.decode('utf-8', errors='ignore')
                if r.matched:
                    t = r.t.decode('utf-8', errors='ignore')
                    regions.append(dict(
                        s=s,
                        t=t,
                        similarity=r.similarity,
                        weight=r.weight,
                        pos_s=r.pos_s.decode('utf-8', errors='ignore'),
                        pos_t=r.pos_t.decode('utf-8', errors='ignore'),
                        metric=r.metric.decode('utf-8', errors='ignore')))
                else:
                    regions.append(dict(s=s, mismatch_penalty=r.mismatch_penalty))

            metadata = m.document.metadata
            speaker, loc_desc = get_location_desc(metadata, m.location)

            matches.append(dict(
                debug=dict(document=m.document.id, sentence=m.sentence_id),
                score=m.score,
                algorithm=m.metric,
                location=dict(
                    speaker=speaker,
                    author=metadata["author"],
                    work=metadata["title"],
                    location=loc_desc
                ),
                regions=regions,
                omitted=m.omitted))
        except UnicodeDecodeError as e:
            traceback.print_exc()

    return matches


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(core)

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
    matches = rs_to_matches(session.find(query))
    with open("/Users/arbeit/Desktop/temp.json", "w") as f:
        f.write(json.dumps(matches, indent=4))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

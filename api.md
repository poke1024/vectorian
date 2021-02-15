import vectorian

# building a corpus
corpus = vectorian.Corpus("/path/to/db")
corpus.docs()
corpus.num_tokens()

doc = vectorian.Document(doc_id, spacy_doc_json, locations)
corpus.add(doc)

im = vectorian.importers.ShakespeareImporter()
corpus.add(im("/path/to/xml"))

im = vectorian.importers.NovelImporter()
corpus.add(im("/path/to/text"))

# file structure
texts/
nlp/
    spacy.3.0.0a/
        text1.json
    spacy.2.0.1b/
        text1.json

files are stored as json
upon opening a Corpus, jsons are converted to pyarrow

# PrecomputedEmbedding
embedding = vectorian.embeddings.FastText("en")

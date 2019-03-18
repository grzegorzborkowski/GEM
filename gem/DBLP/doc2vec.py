from DBLP import DBLP
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import smart_open
import random

dblp = DBLP(100, 1, 10, 3, True)
articles = dblp.read_and_filter_dataset(filterAbstract=False)
dblp.__prepare_external_graph__(articles)
docs = []
indexes = []

for article in articles:
    abstract = article['abstract']
    docs.append(TaggedDocument(words=abstract, tags=[int(article['index_mapped'])]))
    indexes.append(int(article['index_mapped']))


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

result = {}
for idx, doc_id in enumerate(indexes):
    result[doc_id] = model.infer_vector(articles[idx]['abstract'])

print (result)
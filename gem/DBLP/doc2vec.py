from .DBLP import DBLP
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import smart_open
import random

class doc2vec:
    def __init__(self):
        hyper_params = {
            'method_name': 'doc2vec'
        }

    def get_method_name(self):
        return 'doc2vec'

    def learn_embedding(self):
        dblp = DBLP(100, 1, 10, 3, True)
        articles = dblp.read_and_filter_dataset(filterAbstract=False)
        # articles = articles[:50]
        dblp.__prepare_external_graph__(articles)
        docs = []
        indexes = []

        for article in articles:
            abstract = article['abstract']
            docs.append(TaggedDocument(words=abstract, tags=[int(article['index_mapped'])]))
            indexes.append(int(article['index_mapped']))


        model = gensim.models.doc2vec.Doc2Vec(vector_size=2, min_count=2, epochs=40)
        model.build_vocab(docs)
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

        embedding_dict = {}
        for idx, doc_id in enumerate(indexes):
            embedding_dict[doc_id] = model.infer_vector(articles[idx]['abstract'])

        print('doc2vec embeddings learned ' + str(len(embedding_dict)))
        return embedding_dict


if __name__ == '__main__':
    dblp = DBLP(100, 1, 10, 3, True)
    articles = dblp.read_and_filter_dataset(filterAbstract=False)
    dblp.__prepare_external_graph__(articles)
    docs = []
    indexes = []

    for article in articles:
        abstract = article['abstract']
        docs.append(TaggedDocument(words=abstract, tags=[int(article['index_mapped'])]))
        indexes.append(int(article['index_mapped']))


    model = gensim.models.doc2vec.Doc2Vec(vector_size=2, min_count=2, epochs=40)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    result = {}
    for idx, doc_id in enumerate(indexes):
        result[doc_id] = model.infer_vector(articles[idx]['abstract'])
        print(str(doc_id) + ": " + str(result[doc_id]))

    # print(result)
    print (len(result))

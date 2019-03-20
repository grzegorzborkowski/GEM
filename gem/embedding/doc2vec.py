from gem.DBLP.DBLP import DBLP
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import smart_open
import random
from gem.utils import graph_util
from .static_graph_embedding import StaticGraphEmbedding
import numpy as np
from time import time

class doc2vec(StaticGraphEmbedding):
    def __init__(self, *hyper_dict, **kwargs):
        hyper_params = {
            'method_name': 'doc2vec'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        dblp = DBLP(100, 1, 10, 3, True)
        articles = dblp.read_and_filter_dataset(filterAbstract=False)
        docs = []
        embeddings = []
        idx_indexes = {}
        current_idx = 0
        t1 = time()

        for article in articles:
            if int(article['index_mapped']) in graph:
                abstract = article['abstract']
                docs.append(TaggedDocument(words=abstract, tags=[int(article['index_mapped'])]))
                idx_indexes[current_idx] = int(article['index_mapped'])
                current_idx += 1

        vector_size = 2
        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)
        model.build_vocab(docs)
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

        idx_indexes_sorted = sorted(idx_indexes.items(), key=lambda x: x[1])

        current_matrix_id = 0
        for idx, index_mapped in idx_indexes_sorted:
            vector = model.infer_vector(articles[idx]['abstract'])
            #print(str(idx) + " " + str(index_mapped) + " " + str(vector))
            embeddings.append(vector)

        embeddings = np.reshape(embeddings, (-1, vector_size))
        # print('doc2vec embeddings learned ' + str(len(embeddings)))
        #print (embeddings)
        t2 = time()
        self._X = embeddings
        return self._X, t2-t1

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


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
        # print(str(doc_id) + ": " + str(result[doc_id]))

    print (len(result))

from gem.DBLP.DBLP import DBLP
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import smart_open
import random
from gem.utils import graph_util
from .static_graph_embedding import StaticGraphEmbedding
from .doc2vec import doc2vec
from .node2vec import node2vec
import numpy as np
from time import time

class doc2vec_node2vec(StaticGraphEmbedding):
    def __init__(self, *hyper_dict, **kwargs):
        hyper_params = {
            'method_name': 'node2vec_doc2vec'
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
        t1 = time()

        articles = self._articles
        graph = self._graph

        d2v = doc2vec(articles=articles)
        n2v = node2vec(d=2, max_iter=1, walk_len=1, num_walks=1, con_size=1, ret_p=1, inout_p=1, edge_f=edge_f)

        doc2vec_embedding, _ = d2v.learn_embedding(graph=graph)
        node2vec_embedding, _ = n2v.learn_embedding(graph=graph)

        embeddings = []
        new_vector_size = len(doc2vec_embedding[0])+len(node2vec_embedding[0])
        for d2v_vector, n2v_vector in zip(doc2vec_embedding,node2vec_embedding):
            embeddings.append(np.concatenate([d2v_vector,n2v_vector]))
        embeddings = np.reshape(embeddings, (-1, new_vector_size))

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

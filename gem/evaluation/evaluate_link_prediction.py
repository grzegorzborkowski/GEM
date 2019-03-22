try: import cPickle as pickle
except: import pickle
from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util
import numpy as np
import networkx as nx

import sys
sys.path.insert(0, './')
from gem.utils import embed_util

def test():
    print('test!')

def evaluateStaticLinkPrediction(digraph, graph_embedding,
                                 train_ratio=0.8,
                                 n_sample_nodes=None,
                                 sample_ratio_e=None,
                                 no_python=False,
                                 is_undirected=False):
    node_num = digraph.number_of_nodes()
    print('elsp digraph, nodes: ' + str(len(digraph.nodes())))

    # seperate train and test graph
    train_digraph, test_digraph = evaluation_util.splitDiGraphToTrainTest(
        digraph,
        train_ratio=train_ratio,
        is_undirected=is_undirected
    )
    # print('elsp training graph, nodes: ' + str(len(train_digraph.nodes())))

    if not nx.is_connected(train_digraph.to_undirected()):
        train_digraph = max(
            nx.weakly_connected_component_subgraphs(train_digraph),
            key=len
        )
        tdl_nodes = train_digraph.nodes()
        nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
        reversedNodeListMap = dict(zip(range(len(tdl_nodes)),tdl_nodes))
        nx.relabel_nodes(train_digraph, nodeListMap, copy=False)
        test_digraph = test_digraph.subgraph(tdl_nodes)
        nx.relabel_nodes(test_digraph, nodeListMap, copy=False)
    else:
        nodeListMap = dict(zip(tdl_nodes, tdl_nodes))
        reversedNodeListMap = dict(zip(tdl_nodes,tdl_nodes))

    # print('elsp training graph after largest cc, nodes: ' + str(len(train_digraph.nodes())))
    # learning graph embedding

    # doc2vec uses reversedNodeListMap to re-relabel nodes in the given train_digraph
    # in orger to compare these nodes with ids of slef._articles
    if graph_embedding.get_method_name() in ['doc2vec', 'doc2vec_node2vec']:
        X, _ = graph_embedding.learn_embedding(
            resampling_reversed_map=reversedNodeListMap,
            graph=train_digraph,
            no_python=no_python
        )
    else:
        X, _ = graph_embedding.learn_embedding(
            graph=train_digraph,
            no_python=no_python
        )
    node_l = None
    if n_sample_nodes:
        test_digraph, node_l = graph_util.sample_graph(
            test_digraph,
            n_sample_nodes
        )
        X = X[node_l]

    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(
            node_num=node_num,
            sample_ratio=sample_ratio_e,
            is_undirected=is_undirected
        )
    else:
        eval_edge_pairs = None

    estimated_adj = graph_embedding.get_reconstructed_adj(X, node_l)

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        adj=estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )

    # predicted_edge_list = evaluation_util.getEdgeListFromClassifier(
    #     graph_embedding=graph_embedding,
    #     adj=estimated_adj,
    #     full_graph=digraph,
    #     train_graph=train_digraph,
    #     test_graph=test_digraph,
    #     no_python=no_python,
    #     node_list_map=nodeListMap,
    #     reversed_node_list_map=reversedNodeListMap
    # )

    print('predicted_edge_list: ' + str(len(predicted_edge_list)))

    if node_l is None:
        node_l = list(range(train_digraph.number_of_nodes()))

    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(node_l[e[0]], node_l[e[1]])]
    MAP = metrics.computeMAP(filtered_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(
        filtered_edge_list,
        test_digraph
    )
    return (MAP, prec_curv)


def expLP(digraph, graph_embedding,
          n_sample_nodes, rounds,
          res_pre, m_summ, train_ratio=0.8,
          no_python=False, is_undirected=True):
    print('\tLink Prediction')
    summ_file = open('%s_%s.lpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    MAP = [None] * rounds
    prec_curv = [None] * rounds
    for round_id in range(rounds):
        MAP[round_id], prec_curv[round_id] = \
            evaluateStaticLinkPrediction(digraph, graph_embedding,
                                         train_ratio=train_ratio,
                                         n_sample_nodes=1024,
                                         no_python=no_python,
                                         is_undirected=is_undirected)
    summ_file.write('\t%f/%f\t%s\n' % (
        np.mean(MAP),
        np.std(MAP),
        metrics.getPrecisionReport(
            prec_curv[0],
            len(prec_curv[0])
        )
    ))
    summ_file.close()
    pickle.dump([MAP, prec_curv],
                open('%s_%s.lp' % (res_pre, m_summ),
                     'wb'))

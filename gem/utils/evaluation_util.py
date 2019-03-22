import numpy as np
from random import randint
from sklearn.linear_model import LogisticRegression
import networkx as nx


def getRandomEdgePairs(node_num, sample_ratio=0.01, is_undirected=True):
    num_pairs = int(sample_ratio * node_num * (node_num - 1))
    if is_undirected:
        num_pairs = num_pairs / 2
    current_sets = set()
    while(len(current_sets) < num_pairs):
        p = (randint(node_num), randint(node_num))
        if(p in current_sets):
            continue
        if(is_undirected and (p[1], p[0]) in current_sets):
            continue
        current_sets.add(p)
    return list(current_sets)

#
# def getEdgeListFromClassifier(graph_embedding, adj, full_graph, train_graph, test_graph, no_python, \
#  node_list_map, reversed_node_list_map):
#     result = []
#     node_num = adj.shape[0]
#     if graph_embedding.get_method_name() in ['doc2vec', 'doc2vec_node2vec']:
#         tdl_nodes = full_graph.nodes()
#         # nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
#         # reversedNodeListMap = dict(zip(range(len(tdl_nodes)),tdl_nodes))
#         nx.relabel_nodes(full_graph, node_list_map, copy=False)
#         train_node_list_map = {k:v for k,v in node_list_map.items() if k in train_graph.nodes()}
#         nx.relabel_nodes(train_graph, train_node_list_map, copy=False)
#         test_node_list_map = {k:v for k,v in node_list_map.items() if k in test_graph.nodes()}
#         nx.relabel_nodes(test_graph, test_node_list_map, copy=False)
#
#     train_edges = train_graph.edges()[:100]
#     train_nodes = train_graph.nodes()
#     train_edges_false = []
#
#     test_edges = test_graph.edges()
#     test_nodes = test_graph.nodes()
#
#     if graph_embedding.get_method_name() in ['doc2vec', 'doc2vec_node2vec']:
#         X, _ = graph_embedding.learn_embedding(
#             resampling_reversed_map=reversed_node_list_map,
#             graph=full_graph,
#             no_python=no_python
#         )
#     else:
#         X, _ = graph_embedding.learn_embedding(
#             graph=full_graph,
#             no_python=no_python
#         )
#
#     emb_matrix = X
#
#     for i in range(len(train_nodes)):
#         for j in range(len(train_nodes)):
#             if(len(train_edges_false)>=len(train_edges)):
#                 break
#             n1 = train_nodes[i]
#             n2 = train_nodes[j]
#
#             if n1==n2:
#                 pass
#             elif (n1,n2) in train_edges or (n2,n1) in train_edges:
#                 pass
#             else:
#                 train_edges_false.append((min(n1,n2),max(n1,n2)))
#         else:
#             continue  # only executed if the inner loop did NOT break
#         break  # only executed if the inner loop DID break
#
#     train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
#
#     print('positive train edges ' + str(len(train_edges)))
#     print('negatice train edges ' + str(len(train_edges_false)))
#     # Train-set edge embeddings
#     pos_train_edge_embs = get_edge_embeddings(train_edges,emb_matrix)
#     neg_train_edge_embs = get_edge_embeddings(train_edges_false,emb_matrix)
#     train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
#     # Test-set edge embeddings, labels
#     test_edge_embs = get_edge_embeddings(test_edges,emb_matrix)
#     # neg_test_edge_embs = get_edge_embeddings(test_edges_false,emb_matrix)
#     # test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
#
#
#     print('training')
#     # Train logistic regression classifier on train-set edge embeddings
#     edge_classifier = LogisticRegression(random_state=0)
#     edge_classifier.fit(train_edge_embs, train_edge_labels)
#     #
#     # # Predicted edge scores: probability of being of class "1" (real edge)
#     test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
#     print(test_preds)
#     # test_roc = roc_auc_score(test_edge_labels, test_preds)
#     # test_ap = average_precision_score(test_edge_labels, test_preds)
#     print('done')
#
#     for edge, pred in zip(test_edges,test_preds):
#         result.append((edge[0],edge[1],pred))
#
#     # if graph_embedding.get_method_name() in ['doc2vec', 'doc2vec_node2vec']:
#     #     result = [(reversedNodeListMap[e1],reversedNodeListMap[e2],vec) for (e1,e2,vec) in result]
#     print(result[:5])
#     return result

# Generate bootstrapped edge embeddings (as is done in node2vec paper)
# Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list,emb_matrix):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def getEdgeListFromAdjMtx(adj, threshold=0.0, is_undirected=False, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if(j == i):
                    continue
                if(is_undirected and i >= j):
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result

def splitDiGraphToTrainTest(di_graph, train_ratio, is_undirected=True):
    train_digraph = di_graph.copy()
    test_digraph = di_graph.copy()
    node_num = di_graph.number_of_nodes()
    for (st, ed, w) in di_graph.edges_iter(data='weight', default=1):
        if(is_undirected and st >= ed):
            continue
        if(np.random.uniform() <= train_ratio):
            test_digraph.remove_edge(st, ed)
            if(is_undirected):
                test_digraph.remove_edge(ed, st)
        else:
            train_digraph.remove_edge(st, ed)
            if(is_undirected):
                train_digraph.remove_edge(ed, st)
    return (train_digraph, test_digraph)

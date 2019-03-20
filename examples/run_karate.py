'''
Run the graph embedding methods on Karate graph and evaluate them on
graph reconstruction and visualization. Please copy the
gem/data/karate.edgelist to the working directory
'''
import matplotlib.pyplot as plt
from time import time
import pickle
from pathlib import Path

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.evaluation import evaluate_link_prediction as lp

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from argparse import ArgumentParser

from gem.embedding.node2vec             import node2vec
from gem.embedding.doc2vec              import doc2vec
from gem.embedding.doc2vec_node2vec     import doc2vec_node2vec
from gem.DBLP.DBLP                      import DBLP


if __name__ == '__main__':
    ''' Sample usage
    python run_karate.py -node2vec 1 -doc2vec 1 -reload 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Karate Graph')
    parser.add_argument('-node2vec', '--node2vec',
                        help='whether to run node2vec (default: False)')
    parser.add_argument('-doc2vec', '--doc2vec',
                        help='whether to run doc2vec (default: False)')
    parser.add_argument('-doc2vec_node2vec', '--doc2vec_node2vec',
                        help='whether to run doc2vec_node2vec (default: False)')
    parser.add_argument('-reload', '--reload',
                        help='whether to recreate DBLP articles and external graph (default: False)')
    args = vars(parser.parse_args())
    try:
        run_n2v = bool(int(args["node2vec"]))
    except:
        run_n2v = False
    try:
        run_d2v = bool(int(args["doc2vec"]))
    except:
        run_d2v = False
    try:
        run_d2v_n2v = bool(int(args["doc2vec_node2vec"]))
    except:
        run_d2v_n2v = False
    try:
        reload = bool(int(args["reload"]))
    except:
        reload = False

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    #edge_f = 'data/karate.edgelist'
    edge_f = 'external_graph.csv'
    # Specify whether the edges are directed
    isDirected = True

    articles_dump_file = 'articles_dump.pickle'
    if reload or not Path(articles_dump_file).exists():
        dblp = DBLP(100, 1, 10, 3, True)
        articles = dblp.read_and_filter_dataset(filterAbstract=False)
        dblp.__prepare_external_graph__(articles)
        G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=True)
        G = G.to_directed()
        articles = [article for article in articles if int(article['index_mapped']) in G]
        with open(articles_dump_file, 'wb') as handle:
            pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(articles_dump_file, 'rb') as handle:
            articles = pickle.load(handle)
        G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=True)
        G = G.to_directed()

    models = []
    # Load the models you want to run
    # models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0))
    # models.append(HOPE(d=4, beta=0.01))
    # models.append(LaplacianEigenmaps(d=2))
    # models.append(LocallyLinearEmbedding(d=2))
    if run_d2v_n2v:
        models.append(doc2vec_node2vec(articles=articles, graph=G))
    if run_d2v:
        models.append(doc2vec(articles=articles))
    if run_n2v:
        models.append(
            node2vec(d=2, max_iter=1, walk_len=1, num_walks=1, con_size=1, ret_p=1, inout_p=1, edge_f=edge_f)
            # node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1, edge_f=edge_f)
        )
    # models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=100,
                    # modelfile=['enc_model.json', 'dec_model.json'],
                    # weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print('\nProcessing: ' + str(embedding._method_name))
        print('Num articles: %d. Num nodes: %d, num edges: %d' % \
         (len(articles), G.number_of_nodes(), G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        matrix, t = embedding.learn_embedding(graph=G)
        # print(matrix)
        print(embedding._method_name + ' size of embedding matrix: ' + str(len(matrix)))
        # viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        # plt.show()
        # plt.clf()
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        # MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        # ---------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------
        # Visualize
        # viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        # plt.show()
        # plt.clf()
        #print(viz.test())
        MAP, prec_curv = lp.evaluateStaticLinkPrediction(G, embedding, is_undirected=False)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
        print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))

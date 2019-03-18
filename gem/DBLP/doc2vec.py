from DBLP import DBLP
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import smart_open
import random

dblp = DBLP(100,1,10,3,True)
articles = dblp.read_and_filter_dataset(filterAbstract=False)
dblp.__prepare_external_graph__(articles)
docs = []
indexes = []
for article in articles:
    abstract = article['abstract']
    docs.append(TaggedDocument(words=abstract, tags=[int(article['index'])]))
    indexes.append(int(article['index']))


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

for doc_id in indexes:
    print ("doc_id" + str(doc_id))
    idx = indexes[doc_id]
    print ("idx" + str(idx))
    print (model.infer_vector(article[idx]['abstract']))


# ranks = []
# second_ranks = []
# results = {}
# print (docs)
# for doc_id in indexes:
#     print (doc_id)
#     inferred_vector = model.infer_vector(docs[doc_id].words)
#     # print (inferred_vector)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     # print (sims)
#     results[doc_id] = inferred_vector
# print (results)
    # print([docid for docid, sim in sims])
    # print (sims)
    # rank = [docid for docid, sim in sims].index(str(doc_id))
    # ranks.append(rank)

    # second_ranks.append(sims[1])

    # collections.Counter(ranks)

    # # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(docs[doc_id].words)))
    # print('\nDocument ({}): «{}»'.format(doc_id, articles[doc_id]['title']))

    # # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:' % model)
    # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #     # print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(docs[sims[index][0]].words)))
    #     # print ("label" + label)
    #     # print ("index" + index)
    #     # print (type(label))
    #     # print (type(index))
    #     print(u'%s %s: «%s»' % (label, sims[index], articles[sims[index][0]]['title']))
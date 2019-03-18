import csv
import copy
import nltk
import collections
from operator import itemgetter
from nltk.corpus import stopwords
import argparse
import os
import tqdm

class DBLP():

    DEBUG = True

    def __init__(self, authors_count, words_min_frequency, min_edges_for_article, ngrams, debug):
        self.authors_count = authors_count
        self.words_min_frequency = words_min_frequency
        self.min_edges_for_article = min_edges_for_article
        self.ngrams = ngrams
        DBLP.DEBUG = debug
        if DBLP.DEBUG: print ("[DBLP-Pipeline] Checking if stopwords for nltk package are downloaded")
        nltk.download('stopwords')
        if DBLP.DEBUG: print ("[DBLP-Pipeline] NLTK stopwords downloaded")
        self.summary = {}

    def read_and_filter_dataset(self, filterAbstract):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Reading articles from file")
        initial_articles = self.__read_articles__()
        self.summary['initial_articles_count'] = len(initial_articles)

        if DBLP.DEBUG: print ("[DBLP-Pipeline] Initial: Remove articles with few edges")
        initial_articles = self.__remove_articles_with_few_edges(initial_articles)

        if DBLP.DEBUG: print("[DBLP-Pipeline] Reading articles from file has finished. Filtering articles without abstract")
        articles_without_abstract = self.__filter_articles_without_abstract__(initial_articles)

        if DBLP.DEBUG: print("[DBLP-Pipeline] Filtered articles without abstract. Tokenizing and removing stop words from abstract")
        articles_tokenized = self.__tokenize_and_remove_step_words_from_abstract__(articles_without_abstract)

        # if DBLP.DEBUG: print("[DBLP-Pipeline] Finding most frequent authors")
        # most_frequent_authors = self.__find__most_frequent_authors__(articles_tokenized)

        # if DBLP.DEBUG: print("[DBLP-Pipeline] Filtering articles not containing top authors")
        # articles_with_top_authors_only = self.__filter_articles_not_containing_top_authors__(articles_tokenized, most_frequent_authors)

        articles_with_top_authors_only = articles_tokenized

        if DBLP.DEBUG: print("[DBLP-Pipeline] Removing non-existing artciles from quoted-by filed in articles")
        articles_with_updated_quoted_by = self.__update_quoted_by__(articles_with_top_authors_only)
        
        if filterAbstract:
            if DBLP.DEBUG: print("[DBLP-Pipeline] Calculating words frequency")
            top_words = self.__calculate_frequency_of_words__(articles_with_updated_quoted_by)

            if DBLP.DEBUG: print("[DBLP-Pipeline] Building words to number of articles dictionary")
            word_to_number_of_articles = self.__count_number_of_articles_per_word__(articles_with_top_authors_only)

            if DBLP.DEBUG: print("[DBLP-Pipeline] Filtering the most frequent words from abstract")
            articles_with_filtered_abstract = self.__filter_the_most_frequent_words_from_abstract__(articles_with_top_authors_only, word_to_number_of_articles)

            if DBLP.DEBUG: print("[DBLP-Pipeline] Filtering unfrequent words from abstract")
            articles_with_filtered_abstract = self.__filter_unfrequent_words_from_abstract__(articles_with_filtered_abstract, top_words, word_to_number_of_articles)
        else:
            articles_with_filtered_abstract = articles_with_updated_quoted_by

        if DBLP.DEBUG: print("[DBLP-Pipeline] Merging abstract content with title and authors")
        articles_with_merged_content = self.__merge_article_abstract_and_title_authors__(articles_with_filtered_abstract)

        if DBLP.DEBUG: print ("[DBLP-Pipeline] Remove articles with few edges")
        articles_with_sufficient_edges = self.__remove_articles_with_few_edges(articles_with_merged_content)

        if DBLP.DEBUG: print("[DBLP-Pipeline] Removing non-existing artciles from quoted-by filed in articles")
        articles_with_updated_quoted_by = self.__update_quoted_by__(articles_with_sufficient_edges)
        self.summary['articles_count'] = len(articles_with_updated_quoted_by)

        return articles_with_updated_quoted_by

    def prepare_graph_of_graphs_from_articles(self, articles):
        self.__prepare_external_graph__(articles)
        self.__prepare_internal_graph__(articles)

    def __prepare_external_graph__(self, articles):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Writing an external graph content to file")
        with open("external_graph.csv", 'w') as file:
            print (len(articles))
            for article in tqdm.tqdm(articles):
                # print ("XD")
                for quoted_article in article['quoted']:
                    # print ("writing xD")
                    file.write(str(article['index_mapped']) + " " + str(quoted_article) + "\n")
               
    def __prepare_internal_graph__(self, articles):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Writing an internal graph content to files")
        dictionary_of_words_mapping = self.__get_word_mapping_dictionary(articles)
        current_value = 0

        if DBLP.DEBUG: print ("[DBLP-Pipeline] Removing internal_graphs  directory and its content")

        if os.path.exists("internal_graphs"):
            for file_path in os.listdir("internal_graphs"):
                os.remove("internal_graphs/" + file_path)
        if os.path.exists("internal_graphs"): os.removedirs("internal_graphs")

        if DBLP.DEBUG: print ("[DBLP-Pipeline] Creating a internal-graphs directory")

        if not os.path.exists("internal_graphs"):
            os.makedirs("internal_graphs")

        for article in articles:
            file_path = "internal_graphs/" + article["index"] + ".csv"
            with open(file_path, 'w') as file:
                self.__write_internal_graph_for_article__(article, dictionary_of_words_mapping, file_path, file)

    def __write_internal_graph_for_article__(self, article, dictionary_of_words_mapping, file_path, file_descriptor):
        for word_idx in range(len(article['merged_content'])):
            id_of_current_word = dictionary_of_words_mapping[article['merged_content'][word_idx]]
            n_grams_encoding = self.__get_ngrams_for_word__(article, word_idx, self.ngrams, dictionary_of_words_mapping)
            file_descriptor.write(str(dictionary_of_words_mapping[article['merged_content'][word_idx]]) + "," + n_grams_encoding + "\n")
        return

    def __get_ngrams_for_word__(self, article, word_idx, n_grams, dictionary_of_words_mapping):
        n_grams_result = []
        for idx in range(word_idx, word_idx+n_grams):
            if idx < len(article['merged_content']):
                word = article['merged_content'][idx]
                word_mapping = dictionary_of_words_mapping[word]
                n_grams_result.append(str(word_mapping))
        return ",".join(n_grams_result)

    def __get_word_mapping_dictionary(self, articles):
        if DBLP.DEBUG: print ("[DBLP-Pipeline] Preparing a mapping of words to indices")
        dictionary_of_words_mapping = {}
        current_value = 0
        all_words = 0
        unique_words = 0
        for article in articles:
            # print (article)
            for word in article['merged_content']:
                all_words +=1
                if word in dictionary_of_words_mapping: pass
                else:
                    dictionary_of_words_mapping[word] = current_value
                    current_value += 1
                    unique_words +=1

        #print('All words: ' + str(all_words))
        #print('Unique words in selected articles: ' + str(unique_words))
        self.summary['unique_internal_nodes'] = current_value
        #print('self.summary[unique_internal_nodes]' + str(self.summary['unique_internal_nodes']))
        return dictionary_of_words_mapping

    def write_summary_of_dataset(self):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Writing a summary of dataset to summary.csv")
        with open("summary.csv", "w") as file:
            file.write("Initial articles count " + str(self.summary['initial_articles_count']) + "\n")
            file.write("Count of articles after transformations " + str(self.summary['articles_count']) + "\n")
            file.write("Number of words " + str(self.summary['number_of_words']) + "\n")
            file.write("Number of positive edges in graph " + str(self.summary['number_of_positive_edges']) + "\n")
            file.write("Number of unique internal nodes " + str(self.summary['unique_internal_nodes']) + "\n")

    def __read_articles__(self):
        articles = []
        articles_index_map = {}
        current_idx = 0
        with open('DBLP-citation-Jan8.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='#')
            for row in csv_reader:
                if len(row) >= 2:
                    data = row[1]
                    if len(data)<2:
                        continue
                    elif data[:1] == '*':
                        article = {}
                        article['quoted'] = []
                        article['title'] = data[1:]
                    elif data[:1] == '@':
                        article['authors'] = data[1:]
                    elif data[:1] == 't':
                        article['date'] = data[1:]
                    elif data[:1] == 'c':
                        article['source'] = data[1:]
                    elif data[:5] == 'index':
                        idx = int(data[5:])
                        if idx in articles_index_map:
                            pass
                        else:
                            articles_index_map[idx] = current_idx
                            current_idx+=1
                        article['index'] = data[5:]
                        article['index_mapped'] = articles_index_map[idx]
                        articles.append(article)
                    elif data[:1] == '%':
                        # if 'quoted' in article:
                        if len(article['quoted'])>0:
                            #article['quoted'].append(data[1:])
                            if int(data[1:]) in articles_index_map:
                                pass
                            else:
                                articles_index_map[int(data[1:])] = current_idx
                                current_idx+=1
                            idx_mapped = articles_index_map[int(data[1:])]
                            # print (idx_mapped)
                            article['quoted'].append(idx_mapped)
                        else:
                            if int(data[1:]) in articles_index_map:
                                pass
                            else:
                                articles_index_map[int(data[1:])] = current_idx
                                current_idx+=1
                            #article['quoted'].append(data[1:])
                            idx_mapped = articles_index_map[int(data[1:])]
                            # print (idx_mapped)
                            article['quoted'].append(idx_mapped)
                    elif data[:1] == '!':
                        article['abstract'] = data[1:]
                    else:
                        continue
                # print(articles_index_map)
        return articles

    def __filter_articles_without_abstract__(self, articles):

        return [article for article in articles if 'abstract' in article and 'authors' in article]

    def __tokenize_and_remove_step_words_from_abstract__(self, articles):
        tokenizer = nltk.tokenize.SpaceTokenizer()
        stop_words_set = set(stopwords.words('english'))
        articles_copy = copy.deepcopy(articles)
        with_stop_words = 0
        without_stop_words = 0
        for article in articles_copy:
            try:
                article['abstract'] = tokenizer.tokenize(article['abstract'])
                with_stop_words += len(article['abstract'])
                article['abstract'] = [word for word in article['abstract'] if word not in stop_words_set]
                without_stop_words += len(article['abstract'])
            except AttributeError:
                pass

        # print('With stop words: ' + str(with_stop_words))
        # print('Without stop words ' + str(without_stop_words))
        # print('Removed stop words ' + str(with_stop_words-without_stop_words))
        return articles_copy

    def __find__most_frequent_authors__(self, articles):
        top=self.authors_count
        authors_frequency = collections.defaultdict(int)
        for article in articles:
            authors = article['authors'].split(",")
            for author in authors:
                authors_frequency[author]+=1
        # print('All authors ' + str(len(authors_frequency)))
        top_authors = list(sorted(authors_frequency.items(), key=itemgetter(1),reverse=True))
        top_authors = top_authors[:top]
        top_authors_without_occurences = [el[0] for el in top_authors]
        return top_authors_without_occurences

    def __filter_articles_not_containing_top_authors__(self, articles, top_authors):
        def all_authors_in_top(article, top_authors):
            for author in article['authors'].split(","):
                if author not in top_authors:
                    return False
            return True
        return [article for article in articles if all_authors_in_top(article, top_authors)]

    def __calculate_frequency_of_words__(self, articles):
        frequency_cap = self.words_min_frequency
        frequency = collections.defaultdict(int)
        for article in articles:
            for word in article['abstract']:
                frequency[word] += 1

        top_words = list(sorted(frequency.items(), key=itemgetter(1),reverse=True))
        top_words = [word[0] for word in top_words if word[1] >= frequency_cap]
        self.summary['number_of_words'] = len(top_words)
        # print(top_words)
        # print('All words: ' + str(len(frequency)))
        # print('All words with min frequency: ' + str(len(top_words)))
        return top_words

    def __count_number_of_articles_per_word__(self, articles):
        articles_copy = copy.deepcopy(articles)
        word_to_number_of_articles = {}
        for article in articles_copy:
            words_per_article = set()
            for word in article['abstract']:
                words_per_article.add(word)

            for word in words_per_article:
                if word in word_to_number_of_articles:
                    word_to_number_of_articles[word] += 1
                else:
                    word_to_number_of_articles[word] = 1
        return word_to_number_of_articles

    def __filter_the_most_frequent_words_from_abstract__(self, articles, word_to_number_of_articles):
        articles_copy = copy.deepcopy(articles)
        word_in_most_articles = [word for word, occur in word_to_number_of_articles.items() if occur>len(articles_copy)/2]

        previous = 0
        new = 0
        for article in articles_copy:
            previous += len(article['abstract'])
            article['abstract'] = [word for word in article['abstract'] if word not in word_in_most_articles]
            new += len(article['abstract'])

        # print('All articles: ' + str(len(articles)))
        articles_copy = [article for article in articles_copy if len(article['abstract']) > 0]
        # print('Articles containing valid words: ' + str(len(articles_copy)))
        # print('Removed articles containing only the most frequemt words: ' + str(len(articles)-len(articles_copy)))
        # print('Cummulative length of article[abstract] decreased by: ' + str(previous-new))
        return articles_copy

    def __filter_unfrequent_words_from_abstract__(self, articles, top_words, word_to_number_of_articles):
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            article['abstract'] = [word for word in article['abstract'] if word_to_number_of_articles[word]>2 and word in top_words]

        # print('All articles: ' + str(len(articles)))
        articles_copy = [article for article in articles_copy if len(article['abstract']) > 0]
        # print('Articles containing valid frequent words: ' + str(len(articles_copy)))
        # print('Removed articles containing only unfrequent words: ' + str(len(articles)-len(articles_copy)))
        return articles_copy

    def __remove_articles_with_few_edges(self, articles):
        min_edges_cap = self.min_edges_for_article
        # print('All articles ' + str(len(articles)))
        sufficient_edges_articles = [article for article in articles if len(article['quoted']) >= min_edges_cap]
        # print('Articles with sufficient edges ' + str(len(sufficient_edges_articles)))
        return sufficient_edges_articles

    def __update_quoted_by__(self, articles):
        number_of_edges = 0
        all_indexes = set()
        previous = 0
        new = 0
        for article in articles:
            all_indexes.add(article['index_mapped'])
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            previous += len(article['quoted'])
            updated_quoted = [quoted for quoted in article['quoted'] if quoted in all_indexes]
            article['quoted'] = updated_quoted
            new += len(article['quoted'])
            number_of_edges += len(article['quoted'])
        self.summary['number_of_positive_edges'] = number_of_edges

        # print('Quotations of processed articles ' + str(previous))
        # print('Quotations of remaining articles ' + str(new))
        # print('Qotations of removed articles ' + str(previous-new))
        return articles_copy

    def __merge_article_abstract_and_title_authors__(self, articles):
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            # article['merged_content'] = article['authors'].split(",") + article['title'].split(" ") + article['abstract']
            article['merged_content'] = article['abstract']
        return articles_copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authors_count", type=int, default=5000)
    parser.add_argument("--words_min_frequency", type=int, default=300)
    parser.add_argument("--min_edges_for_article", type=int, default=20)
    parser.add_argument("--ngrams", type=int, default=3)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)

    args = parser.parse_args()
    args_params = vars(args)

    authors_count, words_min_frequency, min_edges_for_article, ngrams, debug = args_params['authors_count'], args_params['words_min_frequency'], args_params['min_edges_for_article'], args_params['ngrams'], args_params['debug']
    dblp = DBLP(authors_count=authors_count, words_min_frequency=words_min_frequency, min_edges_for_article=min_edges_for_article, ngrams=ngrams, debug=debug)
    filtered_documents = dblp.read_and_filter_dataset(filterAbstract=True)
    dblp.prepare_graph_of_graphs_from_articles(filtered_documents)
    dblp.write_summary_of_dataset()
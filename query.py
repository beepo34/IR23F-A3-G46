import time
import pickle
import os
import json
import math
import re
from collections import defaultdict, Counter
from index import _tf, _tokenize, Posting
from nltk.stem import PorterStemmer
from nltk.util import bigrams, ngrams
import heapq

dir = os.path.dirname(os.path.abspath(__file__))

def _ranked_search(query: str, index_index, idf_index, id_map):
    # return top documents relating to the query using lnc.ltc weighted cosine similarity
    query_terms = _tokenize(query.split())
    two_gram_terms = [' '.join(two_term) for two_term in bigrams(query_terms)]
    three_gram_terms = [' '.join(three_term) for three_term in ngrams(query_terms, 3)]
    query_tf_weights = _tf(query_terms + two_gram_terms + three_gram_terms)
    length = 0

    scores = defaultdict(float)
    with open(os.path.join(dir, 'index', 'index.pkl'), 'rb') as index:
        terms = []
        for term, query_tf in query_tf_weights.items():
            try:
                term_idf = idf_index[term]
                query_wt = query_tf * term_idf
                terms.append(
                    {
                        'term': term,
                        'idf': term_idf, 
                        'query_wt': query_wt
                    }
                )

                length += math.pow(query_wt, 2)
            except KeyError:
                pass

        # implement early stopping for low query term idf/low posting tf values
        for term_dict in sorted(terms, key=lambda x: x['idf'], reverse = True):
            if term_dict['idf'] < 1:
                break

            index.seek(index_index[term_dict['term']])
            _, postings = pickle.load(index)

            count = 0
            for posting in postings:
                doc_wt = posting.tf
                if count > 2500 and doc_wt < 2.0:
                    break
                scores[posting.id] += term_dict['query_wt'] * doc_wt
                count += 1
    
    query_length = math.sqrt(length)

    heap = []
    # normalize and order the scores
    for id, score in scores.items():
        doc_length = id_map[str(id)]['length']
        normalized_score = score / (query_length * doc_length)
        heapq.heappush(heap, (-normalized_score, id))
    
    results = []
    while heap:
        doc = heapq.heappop(heap)
        results.append(id_map[str(doc[1])]['url'])

    return results


def _get_urls(potential_docs: set[int], docs_tf_idf: dict[int, int], id_map: dict[str, dict]) -> list[int]:
    filtered_docs = {key: value for key, value in docs_tf_idf.items() if key in potential_docs}
    results = sorted(filtered_docs.items(), key=lambda x: x[1], reverse=True)[:5]

    for i, (doc_id, tf_idf) in enumerate(results):
        results[i] = id_map[str(doc_id)]['url']
        print(results[i], ':', tf_idf)

    return results

def _boolean_search(query: str, index_index, id_map) -> list[str]:
    '''Given a query string, will search index for matching documents and return the top 5 urls.'''
    if query == '':
        return []
    
    words = query.split()
    ps = PorterStemmer()
    # get the initial set of documents that contain the first word in the query
    first_word = ps.stem(words[0].lower())

    potential_docs = set()
    docs_tf_idf = defaultdict(int)
    global dir

    with open(os.path.join(dir, 'index', 'index.pkl'), 'rb') as index:
        index.seek(index_index[first_word])
        word, postings = pickle.load(index)
        for posting in postings:
            potential_docs.add(posting.id)
            docs_tf_idf[posting.id] += posting.tf

    if len(words) == 1:
        return _get_urls(potential_docs, docs_tf_idf, id_map)
    
    # for boolean queries, get the intersection of documents that contain each word in the query
    for word in words[1:]:
        word = ps.stem(word.lower())
        docs_for_word = set()
        with open(os.path.join(dir, 'index', 'index.pkl'), 'rb') as index:
            index.seek(index_index[word])
            word, postings = pickle.load(index)
            for posting in postings:
                docs_for_word.add(posting.id)
                docs_tf_idf[posting.id] += posting.tf
        potential_docs = potential_docs.intersection(docs_for_word)

    return _get_urls(potential_docs, docs_tf_idf, id_map)

if __name__ == '__main__':
    print("Starting Search Engine...")

    # load index_index
    with open(os.path.join(dir, 'index', 'index_index.json')) as f:
        index_index = json.load(f)

    # load index_index
    with open(os.path.join(dir, 'index', 'idf_index.json')) as f:
        idf_index = json.load(f)

    # load id_map
    with open(os.path.join(dir, 'index', 'id_map.json')) as f:
        id_map = json.load(f)

    

    while True:
        user_input = input("Enter a query (or 'exit' to quit): ")
        start_time = time.time_ns()
        
        if user_input.lower() == 'exit':
            print("Exiting the Search Engine. Goodbye!")
            break
        
        results = _ranked_search(user_input, index_index, idf_index, id_map)
        end_time = time.time_ns()
        print(f'{len(results)} results ({(end_time - start_time) / 10**6} ms): Showing 15')
        for result in results[:15]:
            print(result)
        print('...')
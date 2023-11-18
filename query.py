import time
import pickle
import os
from collections import defaultdict
from index import Posting
from nltk.stem import PorterStemmer

dir = os.path.dirname(os.path.abspath(__file__))

def boolean_query(query: str) -> list[str]:
    pass

def _get_urls(potential_docs: set[int], docs_tf_idf: dict[int, int], id_map: dict[str, dict]) -> list[int]:
    filtered_docs = {key: value for key, value in docs_tf_idf.items() if key in potential_docs}
    results = sorted(filtered_docs.items(), key=lambda x: x[1], reverse=True)[:5]

    for i, (doc_id, tf_idf) in enumerate(results):
        results[i] = id_map[str(doc_id)]['url']
        print(results[i], ':', tf_idf)

    return results

def search(query: str, index_index, id_map) -> list[str]:
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
    print("Search Engine Start.")

    while True:
        user_input = input("Enter a query (or 'exit' to quit): ")
        start_time = time.time_ns()
        
        if user_input.lower() == 'exit':
            print("Exiting the Search Engine. Goodbye!")
            break
        
        results = boolean_query(user_input)
        end_time = time.time_ns()
        print(f'{len(results)} results ({(end_time - start_time) / 10**6}) ms')
        # TODO: show results
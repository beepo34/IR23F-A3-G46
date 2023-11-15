import re
import json
import pickle
import os
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import heapq
import sys
import math

dir = os.path.dirname(os.path.abspath(__file__))

class Posting(object):
    def __init__(self, id: int, tf: int):
        self.id = id
        self.tf = tf
    
    def __lt__(self, other):
        return self.id < other.id

def _tokenize(phrase_list: list[str]) -> list[str]:
    '''
    Run time is O(n * m)
    '''
    ps = PorterStemmer()
    wordList = []
    for phrase in phrase_list:
        phrase = phrase.strip()
        # includes all alphanumeric in addition to colon and apostrophe
        f = re.findall('[a-zA-Z0-9:\']+', phrase.lower())
        
        # stem words
        wordList += [ps.stem(word) for word in f]

    return wordList


def _tf(tokens: list[str]) -> list:
    return Counter(tokens)

def _weight_tf(term: str, tf: list):
    if tf[term] > 0:
        return math.log(tf[term]) + 1
    else:
        return 0

def _idf(term: str, doc_list: list[list[str]]):
    df = 0
    for doc in doc_list:
        if term in doc:
            df += 1
    if df > 0:
        return math.log(len(doc_list) / df)

def _tfidf(term_list: list[str], doc: list[str], doc_list: list[list[str]]):
    doc_score = 0
    for term in term_list:
        term_weight = _weight_tf(term, _tf(doc))
        term_idf = _idf(term, doc_list)
        term_tfidf = term_weight * term_idf
        doc_score += term_tfidf
    return doc_score


def _load_file(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return {}


def build_index() -> None:
    global dir

    # inverted_index associates tokens with a list of postings
    inverted_index = defaultdict(list)
    id_map = {}
    id = 0
    for subdomain in os.listdir(os.path.join(dir, 'DEV')):
        for file in os.listdir(os.path.join(dir, 'DEV', subdomain)):
            # page: {
            #   'url': str, 
            #   'content': str, 
            #   'encoding': str
            # }
            page = _load_file(os.path.join(dir, 'DEV', subdomain, file))
            if page:
                try:
                    soup = BeautifulSoup(page['content'], 'html.parser')
                    phrase_list = list(soup.stripped_strings)

                    # text in bold, in headings, and in titles should be treated as more important

                    # Add another for bold
                    bold_list = soup.find_all("b")
                    for i in bold_list:
                        phrase_list.append(i.get_text())
                    
                    # Add two more for heading
                    heading_list = soup.find_all(re.compile('^h[1-6]$'))
                    for i in heading_list:
                        heading = i.get_text()
                        phrase_list += [heading, heading]
                    
                    # Add three for title
                    if soup.title:
                        title = soup.title.get_text()
                    phrase_list += [title, title, title]

                    word_list = _tokenize(phrase_list)
                    frequencies = _tf(word_list)

                    for word in frequencies:
                        heapq.heappush(inverted_index[word], Posting(id, frequencies[word]))

                    id_map[id] = {
                        'subdomain': subdomain,
                        'file': file,
                    }
                    
                    id += 1
                except Exception as e:
                    print(e)
            
            # when size of index exceeds 14 MB (~100 MB), offload index to disk
            memory = sys.getsizeof(inverted_index) / 1024**2
            if memory > 14:
                with open(os.path.join(dir, 'indexes', f'index{id}.pkl'), 'wb+') as f:
                    for entry in sorted(inverted_index.items()):
                        pickle.dump(entry, f)
                inverted_index = defaultdict(list)
                
    with open(os.path.join(dir, 'indexes', f'index{id}.pkl'), 'wb+') as f:
        for item in sorted(inverted_index.items()):
            pickle.dump(item, f)

    with open(os.path.join(dir, 'id_map.json'), 'w+') as f:
        json.dump(id_map, f)


def merge_index() -> None:
    # merge partial indexes
    heap = []
    for file in os.listdir(os.path.join(dir, 'indexes')):
        if 'index' in file:
            # open read bufers from partial index files
            partial_index = open(os.path.join(dir, 'indexes', file), 'rb')
            try:
                word, postings =  pickle.load(partial_index)
                heapq.heappush(heap, (word, postings, partial_index))
            except EOFError:
                partial_index.close()

    if not heap:
        return

    # open write buffer to index file
    index = open(os.path.join(dir, 'index.pkl'), 'wb+')
    
    word, postings, partial_index = heapq.heappop(heap)
    cur = (word, postings)
    try:
        word, postings = pickle.load(partial_index)
        heapq.heappush(heap, (word, postings, partial_index))
    except EOFError:
        partial_index.close()
    except ValueError:
        pass
    
    # while partial indexes are not empty, load words into memory
    # and merge postings of the same word
    while heap:
        word, postings, partial_index = heapq.heappop(heap)
        if word == cur[0]:
            # merge postings and store in cur
            cur_postings = cur[1]
            if len(cur_postings) < len(postings):
                while cur_postings:
                    heapq.heappush(postings, heapq.heappop(cur_postings))
                cur = (word, postings)
            else:
                while postings:
                    heapq.heappush(cur_postings, heapq.heappop(postings))
                cur = (word, cur_postings)
            # load the next word from the partial index
            try:
                word, postings = pickle.load(partial_index)
                heapq.heappush(heap, (word, postings, partial_index))
            except EOFError:
                partial_index.close()
            except ValueError:
                pass
        else:
            # merging completed for current word, write to index and update cur
            pickle.dump(cur, index)
            cur = (word, postings)
            # load the next word from the partial index
            try:
                word, postings = pickle.load(partial_index)
                heapq.heappush(heap, (word, postings, partial_index))
            except EOFError:
                partial_index.close()
            except ValueError:
                pass
    
    pickle.dump(cur, index)
    index.close()

    # remove partial index files
    try:
        for file in os.listdir(os.path.join(dir, 'indexes')):
            os.remove(os.path.join(dir, 'indexes', file))
    except OSError:
        print("Error occurred while deleting files.")


def main():
    global dir

    if not os.path.exists(os.path.join(dir, 'DEV')):
        print('DEV folder does not exist, make sure you have unzipped developer.zip into the root folder.')
        return
    
    if not os.path.exists(os.path.join(dir, 'indexes')):
        os.makedirs(os.path.join(dir, 'indexes'))
    
    build_index()
    merge_index()


if __name__ == "__main__":
    main()

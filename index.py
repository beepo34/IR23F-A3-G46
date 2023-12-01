import re
import json
import pickle
import os
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.util import bigrams, ngrams
from urllib.parse import urlparse, urlunparse, urljoin
import heapq
import sys
import math
import hashlib
import pandas as pd

dir = os.path.dirname(os.path.abspath(__file__))
id_map = {}

class Posting(object):
    def __init__(self, id: int, tf: int):
        self.id = id
        self.tf = tf
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id

def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

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
    frequencies = Counter(tokens)
    for term in frequencies:
        frequencies[term] = math.log(frequencies[term]) + 1
    return frequencies


def _idf(n_docs: int, df: int):
    return math.log(n_docs / df)


def _tfidf(tf: float, idf: float):
    return tf * idf


def _load_file(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return {}

def _has_high_textual_content(content) -> bool:
    ''' determines if a page has high textual information content based on the number of paragraphs and number of words '''
    soup = BeautifulSoup(content, 'html.parser')
    # exclude text in footer and menu elements to count
    elements_to_exclude = ['footer', 'nav']

    # Find and remove elements to ignore
    for tag in elements_to_exclude:
        for element in soup.find_all(tag):
            element.extract()

    # count the number of <p> tags
    p_tag_count = len(soup.find_all('p'))

    # get the number of alphanumeric words
    num_words = len(
        [word for word in soup.get_text().split() if word.isalnum()])

    if p_tag_count <= 1 and num_words < 200:
        # classify as not valid because there is not enough textual information
        return False
    else:
        return True
    
def _defragmented_url(url: str) -> str:
    '''Given a url, return the url without its fragments'''
    parsed_url = urlparse(url)

    defragmented_url = urlunparse(parsed_url._replace(fragment=''))
    return defragmented_url

def build_index() -> int:
    global dir, id_map

    # inverted_index associates tokens with a list of postings
    inverted_index = defaultdict(list)
    page_contents = set()
    unique_pages = set()
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

                    # keep track of page urls to check for duplicate pages
                    # if duplicate page, do not add it to the index
                    url_hash = _hash_content(_defragmented_url(page['url']))
                    if url_hash in unique_pages:
                        continue
                    else:
                        unique_pages.add(url_hash)

                    # hash content to check for duplicate pages
                    # if duplicate page, do not add it to the index
                    content_hash = _hash_content(page['content'])
                    if content_hash in page_contents:
                        continue # skip to next file
                    else:
                        page_contents.add(content_hash)

                    # if page does not have high textual content, do not add it to the index
                    if not _has_high_textual_content(page['content']):
                        continue # skip to next file

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

                    terms = _tokenize(phrase_list)
                    tf_weights = _tf(terms)

                    two_gram_terms = bigrams(terms)
                    two_gram_weights = _tf(two_gram_terms)

                    three_gram_terms = ngrams(terms, 3)
                    three_gram_weights = _tf(three_gram_terms)

                    length = 0
                    for term in tf_weights:
                        term_tf = tf_weights[term]
                        length += math.pow(term_tf, 2)
                        heapq.heappush(inverted_index[term], Posting(id, term_tf))

                    # 2 term implementation
                    for two_term in two_gram_weights:
                        two_gram_tf = two_gram_weights[two_term]
                        heapq.heappush(inverted_index[two_term], Posting(id, two_gram_tf))

                    # 3 term implementation
                    for three_term in three_gram_weights:
                        three_gram_tf = three_gram_weights[three_term]
                        heapq.heappush(inverted_index[three_term], Posting(id, three_gram_tf))


                    id_map[id] = {
                        'subdomain': subdomain,
                        'file': file,
                        'url': page['url'],
                        'length': math.sqrt(length)
                    }
                    
                    id += 1
                except Exception as e:
                    print(e)
            
            # when size of index exceeds 14 MB (~100 MB), offload index to disk
            memory = sys.getsizeof(inverted_index) / 1024**2
            if memory > 14:
                with open(os.path.join(dir, 'partial_indexes', f'index{id}.pkl'), 'wb+') as f:
                    for entry in sorted(inverted_index.items()):
                        pickle.dump(entry, f)
                inverted_index = defaultdict(list)
                
    with open(os.path.join(dir, 'partial_indexes', f'index{id}.pkl'), 'wb+') as f:
        for item in sorted(inverted_index.items()):
            pickle.dump(item, f)

    with open(os.path.join(dir, 'index', 'id_map.json'), 'w+') as f:
        json.dump(id_map, f)

    return id


def merge_index(n: int) -> None:
    global id_map

    # merge partial indexes
    heap = []
    for file in os.listdir(os.path.join(dir, 'partial_indexes')):
        if 'index' in file:
            # open read bufers from partial index files
            partial_index = open(os.path.join(dir, 'partial_indexes', file), 'rb')
            try:
                word, postings =  pickle.load(partial_index)
                heapq.heappush(heap, (word, postings, partial_index))
            except EOFError:
                partial_index.close()

    if not heap:
        return

    # open write buffer to index file
    index = open(os.path.join(dir, 'index', 'index.pkl'), 'wb+')
    
    word, postings, partial_index = heapq.heappop(heap)
    cur = (word, postings)
    try:
        word, postings = pickle.load(partial_index)
        heapq.heappush(heap, (word, postings, partial_index))
    except EOFError:
        partial_index.close()
    except ValueError:
        pass
    
    index_index = {}
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
            # merging completed for current word, calculate tfidf, write to index and update cur
            cur_word, cur_postings = cur
            for posting in cur_postings:
                posting.tfidf = _tfidf(posting.tf, _idf(n, len(cur_postings)))

            index_index[cur_word] = index.tell()
            pickle.dump((cur_word, sorted(cur_postings, key=lambda x: (x.tfidf, x.tf), reverse=True)), index)
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

    with open(os.path.join(dir, 'index', 'index_index.json'), 'w+') as f:
        json.dump(index_index, f)

    # remove partial index files and partial_indexes directory
    try:
        for file in os.listdir(os.path.join(dir, 'partial_indexes')):
            os.remove(os.path.join(dir, 'partial_indexes', file))
        os.rmdir(os.path.join(dir, 'partial_indexes'))
    except OSError:
        print("Error occurred while deleting files.")

def main():
    global dir

    if not os.path.exists(os.path.join(dir, 'DEV')):
        print('DEV folder does not exist, make sure you have unzipped developer.zip into the root folder.')
        return
    
    if not os.path.exists(os.path.join(dir, 'partial_indexes')):
        os.makedirs(os.path.join(dir, 'partial_indexes'))

    if not os.path.exists(os.path.join(dir, 'index')):
        os.makedirs(os.path.join(dir, 'index'))

    print('Creating index...')
    
    n = build_index()
    merge_index(n)


if __name__ == "__main__":
    main()
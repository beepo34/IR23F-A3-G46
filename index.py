import re
import json
import os
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

dir = os.path.dirname(os.path.abspath(__file__))

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
        
        # TODO: implement stemming
        wordList += [ps.stem(word) for word in f]

    return wordList


def _tf(tokens: list[str]) -> list:
    return Counter(tokens)


def _load_file(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return {}


def build_index() -> None:
    global dir

    # inverted_index: {
    #   token: [
    #       {
    #           subdomain: str,
    #           id: str,
    #           tf: int,
    #           TODO: add more attributes
    #       },
    #   ],
    # }
    inverted_index = defaultdict(list)
    for subdomain in os.listdir(os.path.join(dir, 'DEV')):
        # TODO: when inverted_index exceeds a threshold size, offload to disk
        
        for file in os.listdir(os.path.join(dir, 'DEV', subdomain)):
            # page: {
            #   'url': str, 
            #   'content': str, 
            #   'encoding': str
            # }
            page = _load_file(os.path.join(dir, 'DEV', subdomain, file))
            if page:
                try:
                    # TODO: ensure the parser can handle broken HTML (missing closing tags, etc.)
                    soup = BeautifulSoup(page['content'], 'html.parser')
                    phrase_list = list(soup.stripped_strings)

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
                    title = soup.title.get_text()
                    phrase_list += [title, title, title]

                    word_list = _tokenize(phrase_list)
                    frequencies = _tf(word_list)

                    # TODO: select important words: text in bold, in headings, 
                    # and in titles should be treated as more important
                    # bold --> x2
                    # headings --> x3
                    # titles --> x4
                    # bold words



                    for word in frequencies:
                        inverted_index[word].append({
                            'subdomain': subdomain,
                            'id': re.sub('.json', '', file),
                            'tf': frequencies[word]
                        })
                except Exception as e:
                    print(e)
    
    with open(os.path.join(dir, 'indexes', 'index.json'), 'w+') as f:
        json.dump(inverted_index, f)


def merge_index():
    # TODO: merge the partial indexes after they have finished being built
    pass


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

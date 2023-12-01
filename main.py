from flask import Flask, request, render_template
import os
import json
import time
from query import _ranked_search
from index import main, Posting
from spellchecker import SpellChecker

app = Flask(__name__, static_url_path='/static')

dir = os.path.dirname(os.path.abspath(__file__))

# load index_index
with open(os.path.join(dir, 'index', 'index_index.json')) as f:
    index_index = json.load(f)

# load index_index
with open(os.path.join(dir, 'index', 'idf_index.json')) as f:
    idf_index = json.load(f)

# load id_map
with open(os.path.join(dir, 'index', 'id_map.json')) as f:
    id_map = json.load(f)

@app.route('/', methods=["GET", "POST"])
def search_query():
    global index_index, id_map
    
    if request.method == "POST":
        try:
            K = 15
            start_time = time.time_ns()
            query = request.form.get("query")
            results_from_query = _ranked_search(query, index_index, idf_index, id_map)
            end_time = time.time_ns()
            first_k_results = results_from_query[:K]
            search_time = (end_time - start_time) / 10**6
            
            return render_template("results.html", query=query, num_results=len(results_from_query), first_k_results=first_k_results, search_time=search_time, K=K)
        except Exception as e:
            print(e)
            # check if user spelled any words wrong
            query = request.form.get("query")
            spell = SpellChecker()
            misspelled = spell.unknown(query.split())
            
            if len(misspelled) == 0:
                return render_template("index.html", error=True, errorMsg="Invalid query. Please try again.")

            try:
                for word in misspelled:
                    # Get the one `most likely` answer
                    query = query.replace(word, spell.correction(word))

                return render_template("index.html", error=True, errorMsg=f"Invalid query. Did you mean: {query}?")
            except Exception as e:
                return render_template("index.html", error=True, errorMsg="Invalid query. Please try again.")
    else:
        return render_template("index.html", error=False, errorMsg="")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import os
import json
from query import search
from index import main, Posting

app = Flask(__name__, static_url_path='/static')

dir = os.path.dirname(os.path.abspath(__file__))
# load index_index
with open(os.path.join(dir, 'index', 'index_index.json')) as f:
    index_index = json.load(f)

# load index_index
with open(os.path.join(dir, 'index', 'id_map.json')) as f:
    id_map = json.load(f)

@app.route('/', methods=["GET", "POST"])
def search_query():
    if request.method == "POST":
        query = request.form.get("query")
        results_from_query = search(query, index_index, id_map)
        
        return render_template("results.html", query=query, results=results_from_query)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
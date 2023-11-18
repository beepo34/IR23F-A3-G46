from flask import Flask, request, render_template
from query import boolean_query

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=["GET", "POST"])
def search_query():
    if request.method == "POST":
        query = request.form.get("query")
        print(query)
        
        results_from_query = boolean_query(query)
        return render_template("results.html", query=query, results=results_from_query)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
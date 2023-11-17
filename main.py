from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=["GET", "POST"])
def search_query():
    if request.method == "POST":
        query = request.form.get("query")
        print(query)
        # search for query here
        results_from_query = ["https://ics.uci.edu/", "https://cs.ics.uci.edu/", "informatics.uci.edu/", "stat.uci.edu/"]
        return render_template("results.html", query=query, results=results_from_query)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
from src import Index, Retrieval, ReRanking
from flask import Flask, request, flash, render_template
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)

index = Index()
retriever = Retrieval(index)
reranker = ReRanking(retriever)


@app.get('/retrieve/<string:qexp>')
def retrieve(qexp) -> str:
    """retrieve search results using either rm3 or splade query expansion"""
    num = request.args.get("num")
    query = request.args.get("q")
    if query is None:
        return "invalid or empty query", 400

    response: str = retriever.search(query, num, qexp).to_json(orient="index")
    return response, 200


@app.get('/search/<string:qexp>')
def rerank(qexp) -> str:
    """retrieve (bm25 + rm3) and rerank (monoT5) search results"""
    num = request.args.get("num")
    query = request.args.get("q")
    if query is None:
        return "invalid or empty query", 400

    response: str = reranker.search(query, num, qexp).to_json(orient="index")
    return response, 200


@app.route('/', methods=("GET", "POST"))
def search():
    """search engine with a GUI"""
    if request.method == "POST":
        query = request.form["query"]
        model = request.form["model"]
        if not query or not model:
            flash("Query and Model is required.")
        else:
            if model == "bm25_rm3":
                results = retriever.search(query, 100, "rm3")
            elif model == "bm25_splade":
                results = retriever.search(query, 100, "splade")
            elif model == "bm25_rm3_t5":
                results = reranker.search(query, 100, "rm3")
            elif model == "bm25_splade_t5":
                results = reranker.search(query, 100, "splade")
            else:
                flash('invalid model')
            return render_template("search.html", results=results.to_dict(orient="records"))
    
    return render_template("search.html", results=None)


if __name__ == "__main__":
    app.run(port=8000)
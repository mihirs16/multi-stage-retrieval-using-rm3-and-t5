from src import Index, Retrieval, ReRanking
from flask import Flask, request

app = Flask(__name__)
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


@app.get('/search')
def rerank() -> str:
    """retrieve (bm25 + rm3) and rerank (monoT5) search results"""
    num = request.args.get("num")
    query = request.args.get("q")
    if query is None:
        return "invalid or empty query", 400

    response: str = reranker.search(query, num).to_json(orient="index")
    return response, 200
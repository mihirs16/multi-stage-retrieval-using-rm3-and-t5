from src import Retrieval
from flask import Flask, request

app = Flask(__name__)
retriever = Retrieval()

@app.get('/retrieve/<string:qexp>')
def retrieve(qexp) -> str:
    """retrieve search results using either rm3 or splade query expansion"""
    num = request.args.get("num")
    query = request.args.get("q")
    if query is None:
        return "invalid or empty query", 400

    response: str = retriever.search(query, num, qexp).to_json(orient="index")
    return response, 200

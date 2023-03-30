from src import Retrieval
from flask import Flask, request
import json

app = Flask(__name__)
bm25_rm3 = Retrieval(query_expansion="rm3")
bm25_splade = Retrieval(query_expansion="splade")

@app.get('/retrieve/<string:qexp>')
def retrieve(qexp) -> str:
    """retrieve search results using either rm3 or splade query expansion"""
    num = request.args.get("num")
    query = request.args.get("q")
    if query is None:
        return "invalid or empty query"

    response: str
    if qexp == "rm3":
        if num is not None:
            response = json.dumps(bm25_rm3.search(query, int(num)))
        else:
            response = json.dumps(bm25_rm3.search(query))
    elif qexp == "splade":
        if num is not None:
            response = json.dumps(bm25_splade.search(query, int(num)))
        else:
            response = json.dumps(bm25_splade.search(query))
    else:
        return "invalid query expansion model, use `rm3` or `splade`."
    
    return response

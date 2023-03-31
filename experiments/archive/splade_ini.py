from pyserini.encode.query import SpladeQueryEncoder
from pyserini.search.lucene import LuceneSearcher, querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
from ir_measures import ScoredDoc, read_trec_qrels,calc_aggregate, AP, nDCG, R
import json

# lucene analyzer for query cleaning
analyzer = Analyzer(get_lucene_analyzer())

# init splade
splade = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil")

# set bm25
ranker = LuceneSearcher('index/CODEC/')
ranker.set_bm25(k1=2.5, b=0.6)

# import topics
queries = []
qids    = []
topics = json.load(open('CODEC/topics/topics.json'))
qids = list(topics.keys())
for qid in qids:
    queries.append(topics[qid]['Query'])

def encode_query(raw_query: str):
    """to encode a query using SPLADE"""
    encoded_query = splade.encode(raw_query)

    query_terms = []
    for k, v in encoded_query.items():
        if len(analyzer.analyze(k)) > 0:
            query_terms.append(
                querybuilder.get_boost_query(
                    querybuilder.get_term_query(k), v
                )
            )

    should_val = querybuilder.JBooleanClauseOccur['should'].value
    boolean_qb = querybuilder.get_boolean_query_builder()
    for query_term in query_terms:
        boolean_qb.add(query_term, should_val)

    return boolean_qb.build()


# compile a run
enc_queries = [encode_query(query) for query in queries]
run = []
for i in range(len(qids)):
    results = ranker.search(enc_queries[i], k=1000)
    for result in results:
        run.append(ScoredDoc(qids[i], result.docid, result.score))


# read qrels
qrels = read_trec_qrels('CODEC/qrels/document_ndcg.qrels')


# measure
metrics = calc_aggregate([R@1000], qrels, run)
print('splade + bm25 -> ', metrics)
from pyserini.search.lucene import LuceneSearcher
from ir_measures import ScoredDoc, read_trec_qrels,calc_aggregate, AP, nDCG, R
import json


# set bm25 + rm3
ranker = LuceneSearcher('index/CODEC/')
ranker.set_bm25(k1=2.5, b=0.6)
ranker.set_rm3(fb_terms=95, fb_docs=20, original_query_weight=0.6)


# import topics
queries = []
qids    = []
topics = json.load(open('CODEC/topics/topics.json'))
qids = list(topics.keys())
for qid in qids:
    queries.append(topics[qid]['Query'])


# batch search for all topics
results = ranker.batch_search(
    queries = queries,
    qids = qids,
    k = 1000,
    threads = 4
)


# compile a run
run = []
for qid in results.keys():
    for result in results[qid]:
        run.append(ScoredDoc(qid, result.docid, result.score))


# read qrels
qrels = read_trec_qrels('CODEC/qrels/document_ndcg.qrels')


# measure
metrics = calc_aggregate([R@1000], qrels, run)
print('bm25 + rm3 -> ', metrics)
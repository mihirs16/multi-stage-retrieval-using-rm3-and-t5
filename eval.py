from src import CODEC, Index, Retrieval, ReRanking
import pyterrier as pt
from pyterrier.measures import MAP, R, nDCG

index = Index()
retrieval = Retrieval(index)
reranker = ReRanking(retrieval)

# fine-tuned BM25
bm25 = retrieval.tokenise >> retrieval.bm25

# fine-tuned BM25 + RM3
bm25_rm3 = retrieval.tokenise >> retrieval.bm25 >> retrieval.rm3 >> retrieval.bm25

#  BM25 + zero-shot SPLADE
bm25_splade =  retrieval.splade.query() >> retrieval.bm25

# fine-tuned BM25 + RM3 + T5
bm25_rm3_t5 = bm25_rm3 >> pt.rewrite.reset() \
    >> pt.text.get_text(retrieval.index, 'text') \
    >> reranker.sliding_window >> reranker.monoT5 \
    >> pt.text.max_passage()

# run the experiments
experiment = pt.Experiment(
    retr_systems = [bm25, bm25_rm3, bm25_splade, bm25_rm3_t5],
    names        = ["bm25", "bm25_rm3", "bm25_splade", "bm25_rm3_t5"],
    topics       = CODEC.topics(),
    qrels        = CODEC.qrels(),
    eval_metrics = [MAP, R@10, R@1000, nDCG@10, nDCG@1000],
    save_dir     = './results/',
    verbose      = True
)

# show evaluation metrics and save them to csv
print(experiment.head())
experiment.to_csv("./results/metrics.csv")
print("----> Experiment Done <----")
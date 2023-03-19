import pyterrier as pt; pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from pyterrier.batchretrieve import BatchRetrieve
from pyterrier.measures import MAP, AP, R, nDCG

from pyterrier_t5 import MonoT5ReRanker
import pyt_splade

import codec_se

import numpy as np

# load index
index = pt.IndexRef.of("./index/data.properties")

# pipeline utils
tokenise = pt.rewrite.tokenise()

# define a tf-idf retrieval
tfidf = BatchRetrieve(index, wmodel="TF_IDF")

# define the bm25 model
bm25 = BatchRetrieve(index, wmodel="BM25")

# tuned bm25
bm25_tuned = BatchRetrieve(
    index_location=index, 
    wmodel="BM25", 
    controls={"bm25.b" : 0.3, "bm25.k_1": 3.5, "bm25.k_3": 4.7 },
)

# bm25 + rm3
rm3_tuned = pt.rewrite.RM3(index, fb_terms=6.0, fb_docs=15.0)
bm25_rm3_tuned = bm25_tuned >> rm3_tuned >> bm25_tuned

# splade query expansion
splade = pyt_splade.SpladeFactory(agg="max").query()

# monoT5 re-ranker
monoT5 = MonoT5ReRanker()

# plug and play tuner
# g_search = pt.GridSearch(
#     splade >> bm25_tuned,
#     { 
#         bm25_tuned : { 
#             "bm25.b" : np.arange(0.1, 1.0, 0.1),
#             "bm25.k_1": np.arange(0.1, 5.0, 0.2),
#             "bm25.k_3": np.arange(0.1, 5.0, 0.2),
#         },
#         # rm3_tuned: {
#         #     "fb_terms": np.arange(5.0, 95.0, 1.0),
#         #     "fb_docs": np.arange(5.0, 20.0, 5.0)
#         # }
#     },
#     topics=codec_se.get_topics(),
#     qrels=codec_se.get_qrels(),
#     metric="map",
#     verbose=True
# )
# print("---> TUNING DONE <---")

# run experiments and eval
experiment = pt.Experiment(
    [
        tokenise >> bm25, 
        tokenise >> bm25_tuned, 
        tokenise >> bm25_rm3_tuned,
        splade   >> tfidf,
        # tokenise >> bm25_rm3_tuned % 100 >> pt.text.get_text(index, 'text') >> monoT5,
        # splade   >> tfidf           >> pt.text.get_text(index, "text") >> monoT5
        # g_search
    ],
    codec_se.get_topics(),
    codec_se.get_qrels(),
    eval_metrics=[MAP, R@100, nDCG@100, nDCG@10],
    verbose=True
)
print(experiment.head())
# experiment['name'] = ["bm25", "bm25_tuned", "bm25_rm3_tuned", "splade_tfidf", "bm25_rm3_monoT5", "splade_tfidf_monoT5"]
# experiment.to_csv('results.csv')
print("---> EXPERIMENT DONE <---")




# ---- archive -----
# bm25_splade = pyt_splade.SpladeFactory(agg="max").query() >> bm25
# bm25_rm3 = bm25 >> pt.rewrite.RM3(index) >> bm25
# bm25_monoT5 = bm25 >> monoT5

# single query search from stored index
# results = bm25.search("How has the UK Open Banking Regulation benefited challenger banks?")
# print(results.head())
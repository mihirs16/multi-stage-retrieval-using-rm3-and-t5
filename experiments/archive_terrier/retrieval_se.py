import pyterrier as pt
pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

from pyterrier.batchretrieve import BatchRetrieve
from pyterrier.measures import MAP, R, nDCG
from pyterrier_t5 import MonoT5ReRanker

from experiments import codec_se
from experiments import CustomSlidingWindow


# load index
index = pt.IndexRef.of("./index/data.properties")

# pipeline utils
tokenise = pt.rewrite.tokenise()

# tuned bm25
bm25_tuned = BatchRetrieve(
    index_location  = index, 
    wmodel          = "BM25", 
    controls        = {
        "bm25.b" : 0.6, 
        "bm25.k_1": 2.5,
        "bm25.k_3": 4.9
    },
) % 100

# bm25 + rm3
rm3_tuned = pt.rewrite.RM3(index, fb_terms=95, fb_docs=20)
bm25_rm3_tuned = bm25_tuned >> rm3_tuned >> bm25_tuned

# monoT5 re-ranker
monoT5 = MonoT5ReRanker(
    model='castorini/monot5-base-msmarco-10k', 
    verbose=True
)
t5_window = CustomSlidingWindow.custom_sliding(
    text_attr    = 'text', 
    length       = 512, 
    stride       = 256, 
    prepend_attr = None,
)
t5_pipe = pt.text.get_text(index, 'text') \
    >> t5_window >> monoT5 >> pt.text.max_passage()

# run experiments and eval
experiment = pt.Experiment(
    retr_systems = [
        tokenise >> bm25_tuned, 
        tokenise >> bm25_rm3_tuned,
        tokenise >> bm25_tuned >> t5_pipe,
        # tokenise >> bm25_rm3_tuned >> t5_pipe,
    ],
    names        = [
        'bm25', 
        'bm25_rm3',
        'bm25_t5',
        # 'bm25_rm3_t5',ż
    ],
    topics       = codec_se.get_topics(),
    qrels        = codec_se.get_qrels(),
    eval_metrics = [MAP, R@100, nDCG@100,  nDCG@10],
    save_dir     = './experiments/results/',
    # save_mode    = 'overwrite',
    verbose      = True,
)
print(experiment.head())
print("---> EXPERIMENT DONE <---")

# # search single query
# engine = tokenise >> bm25_tuned >> pt.text.get_text(index, 'text')
# results = engine.search("How has the UK's Open Banking Regulation benefited challenger banks?")
# print(results[['docno', 'text']].head())
# # t5_tok = CustomSlidingWindow.AutoTokenizer.from_pretrained('t5-base')
# # print(len(t5_tok.tokenize(results['text'].values[4])))
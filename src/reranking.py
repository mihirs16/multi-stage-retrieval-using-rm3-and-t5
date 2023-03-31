import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
from .retrieval import Retrieval

class ReRanking:
    """class to expose the re-ranking models

    attrs:
        retrieval: `Retrieval` object for first-stage
        monoT5: re-ranking transformer model
        sliding_window: split long docs into passages
    """

    def __init__(
            self,
            retrieval: Retrieval,
            t5_model: str = "castorini/monot5-base-msmarco-10k",
            tokeniser: str = "t5-base",
            batch_size: int = 12,
            passage_window: int = 512,
            passage_stride: int = 216
        ):
        """
            initialize the retrieval models for first-stage
            and the monoT5 model for second-stage ranking.
            the monoT5 model has a limit of 512 tokens, thus we
            use a sliding window of 512 and steps of 216 to 
            consider a set of passages for each document and take
            the maximum score of a passage for each document.

            args:
                retrieval: Retrieval models object
                t5_model: which t5 model to use for ranking
                tokeniser: which HF model to use for tokeniser
                batch_size: batch size to rank using `t5_model`
                passage_window: tokens in a passage of a document

        """
        self.retrieval = retrieval
        self.monoT5 = MonoT5ReRanker(
            tok_model=tokeniser,
            model=t5_model,
            batch_size=batch_size
        )
        self.sliding_window = pt.text.sliding(
            text_attr='text',
            length=passage_window,
            stride=passage_stride,
            prepend_attr=None 
        )

    def search(
            self, 
            query: str,  
            num_results: int | str
        ):
        """first-stage retrieval and second-stage ranking for a query

        args:
            query: `str`
            num_results: number of results to show
        """
        num_results = 100 if num_results is None else int(num_results)
        bm25 = self.retrieval.bm25 % num_results

        # stage 1 - BM25 + RM3
        stage_1 = self.retrieval.tokenise >> bm25 \
            >> self.retrieval.rm3 >> bm25
        
        # stage 2 - sliding_window + monoT5
        stage_2 = pt.text.get_text(self.retrieval.index, 'text') \
            >> self.sliding_window >> self.monoT5 \
            >> pt.text.max_passage()
        
        # end-to-end search engine
        engine = stage_1 >> pt.rewrite.reset() >> stage_2 >> pt.rewrite.reset()
        return engine.search(query)

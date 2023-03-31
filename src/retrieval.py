import pyterrier as pt
from pyterrier.batchretrieve import BatchRetrieve
from pyt_splade import SpladeFactory
from typing import Literal

class Retrieval:
    """class for exposing retrieval models

    attrs:
        index_dir: path to index directory
        model: bm25 model fine-tuned for CODEC
        query_expansion: `rm3` or `splade`
        splade_model: [optional] huggingface model type for SPLADE  
    """

    def __init__(
            self, 
            index_path: str = "./index/data.properties",
            splade_model: str = "naver/splade-cocondenser-ensembledistil",
        ):
        """init retrieval model
        
        args:
            index_dir: `str`
            query_expansion: "rm3" or "splade"
        """
        self.index = pt.IndexRef.of(index_path)
        self.tokenise = pt.rewrite.tokenise()
        self.bm25 = BatchRetrieve(
            index_location=self.index,
            wmodel="BM25",
            controls={
                "bm25.b": 0.6,
                "bm25.k_1": 2.5
            }
        )
        self.rm3 = pt.rewrite.RM3(
            self.index,
            fb_terms=95,
            fb_docs=20,
            fb_lambda=0.6
        )
        self.splade = SpladeFactory(model=splade_model)
        self.reset_query = pt.rewrite.reset()
    
    def search(
            self, 
            query: str,  
            num_results: int | str,
            expansion: None | Literal["rm3", "splade"] = None,
        ):
        """initiate a bm25 search for the given query
        
        args:
            query: `str`
            num_results: number of results to return (default = 100) 
            expansion: "rm3" or "splade" query expansion model
        
        returns:
            search results pandas.DataFrame 
        """
        # default number of results 
        num_results = 100 if num_results is None else int(num_results)
        bm25 = self.bm25 % num_results

        # init engine
        engine = self.tokenise

        # build search engine with bm25 and query expansion
        if expansion == "rm3":
            engine = engine >> bm25 >> self.rm3 >> bm25 >> self.reset_query >> self.reset_query
        elif expansion == "splade":
            engine = self.splade.query() >> bm25 >> self.reset_query
        else:
            engine = engine >> bm25

        engine = engine >> pt.text.get_text(self.index, "text")
        return engine.search(query)

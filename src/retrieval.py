from pyserini.search.lucene import LuceneSearcher, querybuilder
from pyserini.encode.query import SpladeQueryEncoder
from pyserini.analysis import Analyzer, get_lucene_analyzer
from typing import Literal
import json

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
            query_expansion: Literal["rm3", "splade"],
            index_dir: str = "./index/CODEC/",
            splade_model: str = "naver/splade-cocondenser-ensembledistil" 
        ):
        """init retrieval model
        
        args:
            index_dir: `str`
            query_expansion: "rm3" or "splade"
        """
        self.index_dir = index_dir
        
        self.model = LuceneSearcher(self.index_dir)
        self.model.set_bm25(k1 = 2.5, b = 0.6)

        self.query_expansion = query_expansion
        if self.query_expansion == "rm3":
            self.model.set_rm3(
                fb_terms = 95, 
                fb_docs = 20, 
                original_query_weight = 0.6
            )
        elif self.query_expansion == "splade":
            self.analyzer = Analyzer(get_lucene_analyzer())
            self.splade = SpladeQueryEncoder(splade_model)
        else:
            raise Exception("invalid query expansion method, use `rm3` or `splade`.")
    
    def encode_query(self, raw_query: str):
        """encode a query using SPLADE
        
        args:
            raw_query: `str`
        
        returns:
            encoded query in the form of a 
            boolean query builder with boosted weights
            according to SPLADE expanded terms
        """
        encoded_query = self.splade.encode(raw_query)

        query_terms = []
        for term, weight in encoded_query.items():
            if len(self.analyzer.analyze(term)) > 0:
                query_terms.append(
                    querybuilder.get_boost_query(
                        querybuilder.get_term_query(term), weight
                    )
                )
        
        should_val = querybuilder.JBooleanClauseOccur['should'].value
        boolean_qb = querybuilder.get_boolean_query_builder()
        for query_term in query_terms:
            boolean_qb.add(query_term, should_val)

        return boolean_qb.build()
    
    def search(self, query: str, num_results: int = 100, threads: int = 4) -> list[dict]:
        """initiate a search for the given query
        
        args:
            query: `str`
            num_results: number of results to return
        
        returns:
            list of each document as a dict
        """
        query = self.encode_query(query) if self.query_expansion == "splade" else query
        hits = self.model.search(query, num_results)
        docs = self.model.batch_doc(
            docids=[hit.docid for hit in hits], 
            threads=threads
        )
        
        results = list()
        for hit in hits:
            results.append({
                    'docid': hit.docid,
                    'score': hit.score, 
                    'content': json.loads(docs[hit.docid].raw())
            })
        
        return results
    
    def batch_search(
            self, 
            queries: list[str], 
            qids: list[str], 
            num_results: int = 100,
            threads: int = 4
        
        ) -> dict[str, list[dict]]:
        """initiate a batch search for the given query
        
        args:
            queries: list of `str` queries
            qids: list of `str` qids
            num_results: number of results to return for each query

        returns:
            dict of each qid and list of each result document as a dict 
        """
        results = dict()
        if self.query_expansion == "rm3":
            hits = self.model.batch_search(queries, qids, num_results, threads)

            for qid in hits.keys():
                results[qid] = []
                for hit in hits[qid]:
                    results[qid].append({
                        hit.docid: hit.score
                    })
        else:
            for i in range(len(qids)):
                results[qids[i]] = self.search(queries[i], num_results)
        
        return results


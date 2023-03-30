import json
import os

class CODEC:
    """class for working with the CODEC repository.

    attrs:
        root_dir: path to root dir of the local CODEC repo.
        corpus_dir: path to the actual documents (jsonl) file.  
    """

    def __init__(
            self, 
            root_dir: str   = "CODEC/", 
            corpus_dir: str = "CODEC/corpus/raw",
        ):
        """Initialize directory paths to work with
        
        args:
            root_dir: `str` root directory of CODEC repo
            corpus_dir: `str` directory of corpus (jsonl) files

        """
        self.root_dir   = root_dir
        self.corpus_dir = corpus_dir

    def load_topics(
            self,
            topics_path: str = "/topics/topics.json"
        ) -> tuple[list, list]:
        """Load topics from specified or default topics file
        
        args:
            topics_dir: `str` directory under the {root_dir}

        returns:
            Tuple(List[qid], List[query])
        """
        topics_dir = os.path.join(self.root_dir, topics_path)

        queries = []
        qids    = []
        topics = json.load(open(topics_dir))
        qids = list(topics.keys())
        for qid in qids:
            queries.append(topics[qid]['Query'])

        return qids, queries
    
    def load_qrel(
            self,
            qrels_path: str = "/qrels/document_ndcg.qrels"
    ) -> str:
        """returns the default qrels path for `read_trec_qrels()`"""
        return os.path.join(self.root_dir, qrels_path)
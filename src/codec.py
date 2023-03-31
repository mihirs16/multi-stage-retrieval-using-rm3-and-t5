import pandas as pd
import json
from typing import Generator

class CODEC:
    """class for working with the CODEC repository.

    attrs:
        root_dir: path to root dir of the local CODEC repo.
        corpus_dir: path to the actual documents (jsonl) file.  
    """

    @staticmethod
    def topics(path: str = "CODEC/topics/topics.json") -> pd.DataFrame:
        """load topics from specified or default topics.json file
        
        args:
            topics_path: `str`

        returns:
            pandas.DataFrame
        """
        df_topics = pd.read_json(path, orient="index").reset_index()
        df_topics = df_topics.drop(["Domain", "Guidelines"], axis=1)
        df_topics.columns = ["qid", "query"]
        return df_topics
    
    @staticmethod
    def qrels(path: str = "CODEC/qrels/document_ndcg.qrels") -> pd.DataFrame:
        """load qrels from specified or default *_ndcg.qrels file
        
        args:
            topics_path: `str`

        returns:
            pandas.DataFrame
        """
        df_qrels = pd.read_csv(
            path, 
            sep=" ", 
            names=["qid", "ignore", "docno", "label"]
        )
        return df_qrels
    
    @staticmethod
    def corpus(path: str = "CODEC/corpus/codec_documents.jsonl") -> Generator:
        """load corpus as a generator of each document
        
        args:
            path: `str` 

        returns:
            dict generator
        """
        with open(path, 'rt') as file:
            for each_row in file:
                each_doc = json.loads(each_row)
                each_doc['docno'] = each_doc.pop('id')
                each_doc['text'] = each_doc.pop('contents')
                yield each_doc

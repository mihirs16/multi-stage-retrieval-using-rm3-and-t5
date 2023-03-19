import re
import json
import pandas as pd

def iter_file(filename):
    """
    load jsonl as a generator
    """
    with open(filename, 'rt') as file:
        for each_row in file:
            each_row_data = json.loads(each_row)
            each_row_data['docno'] = each_row_data.pop('id')
            each_row_data['text'] = each_row_data.pop('contents')
            yield each_row_data

def get_topics(filename='CODEC/topics/topics.json'):
    """
    load topics.json as a dataframe
    """
    df_topics = pd.read_json(filename, orient='index').reset_index().drop(['Domain', 'Guidelines'], axis=1)
    df_topics.columns = ['qid', 'query']
    return df_topics


def get_qrels(filename='CODEC/qrels/document_ndcg.qrels'):
    """
    load qrels as csv
    """
    df_qrels = pd.read_csv(filename, sep=" ", names=["qid", "ignore", "docno", "label"])
    return df_qrels


def clean_query(q_object):
    """
    takes a query and cleans it
    """
    regex = r"[^\s\w\d]"
    return re.sub(regex, " ", q_object['query'], 0)
    
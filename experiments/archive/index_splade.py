from pyserini.encode import SpladeQueryEncoder
import pandas as pd

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def index_w_splade(
    input_path  ='CODEC/corpus/raw/codec_documents.jsonl',
    output_dir ='CODEC/corpus/splade_expanded',
    splade_model='naver/splade-cocondenser-ensembledistil',
    chunksize   = 10 ** 2,
    threads     = 8
):

    def save_expanded(chunk):
        def index_expanded(doc):
            expanded_doc = splade.encode(doc)
            expanded_doc = doc + " " + " ".join(list(expanded_doc.keys()))
            return expanded_doc

        chunk['contents'] = chunk.apply(lambda row: index_expanded(row['contents']), axis=1)
        output_path = f'{output_dir}/{threading.get_ident()}_codec_splade.csv'
        chunk.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        return f'{threading.get_ident()} - completed'

    splade = SpladeQueryEncoder(splade_model)
    reader = pd.read_json(input_path, lines=True, chunksize=chunksize)
    future = dict()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for chunk in reader:
            future[executor.submit(save_expanded, chunk)] = (chunk['id'].values[0], chunk['id'].values[-1])
        for completed in as_completed(future):
            chunk_coord = future[completed]
            print(f'{chunk_coord[0]} to {chunk_coord[1]} saved by {completed.result()}')
            
if __name__ == '__main__':
    index_w_splade(splade_model='naver/efficient-splade-V-large-doc')
    

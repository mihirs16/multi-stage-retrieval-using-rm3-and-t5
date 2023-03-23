import pyterrier as pt; pt.init()
from pyterrier.index import IterDictIndexer

from codec_se import iter_file

indexer = IterDictIndexer(
    index_path='./index',
    meta={'docno': 32, 'text': 4096},
    overwrite=True,
    verbose=True
)
indexref = indexer.index(iter_file('CODEC/corpus/codec_documents.jsonl'))



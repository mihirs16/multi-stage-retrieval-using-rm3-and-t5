import pyterrier as pt
from pyterrier.index import IterDictIndexer
from .codec import CODEC
import os

class Index:
    """class for creating a index from corpus
        or retrieving the index object for searching

    attrs:
        dir: path to index directory
        ref: reference to the index object
    """

    def __init__(self, index_dir: str = './index/'):
        """
            create a new index from CODEC and 
            save the reference to the index

            args:
                index_dir: `str` directory of index
        """
        
        # create index_dir if it doesn't exist
        if not os.path.isdir(index_dir):
            print(f"Creating {index_dir}")
            os.mkdir(index_dir)
        self.dir = index_dir

        # if index already exists
        index_properties_file = os.path.join(self.dir, "data.properties")
        if os.path.isfile(index_properties_file):
            print(f"Loading index at {self.dir}")
            self.ref = pt.IndexRef.of(index_properties_file)
        else:
            print(f"Creating index at {self.dir}")
            self.ref = IterDictIndexer(
                index_path=self.dir,
                meta={ "docno": 32, "text": 6144 },
                verbose=True
            ).index(CODEC.corpus())

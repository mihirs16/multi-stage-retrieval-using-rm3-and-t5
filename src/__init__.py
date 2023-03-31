import pyterrier
pyterrier.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

from .codec     import CODEC
from .index     import Index
from .retrieval import Retrieval
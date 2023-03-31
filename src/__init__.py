import pyterrier
pyterrier.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

from .retrieval import Retrieval
from .codec     import CODEC
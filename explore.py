from src import Index
import pyterrier as pt
import pandas as pd

# load index as explorable
index = pt.IndexFactory.of(Index().ref)

# show overall statistics
print("Index Statistics")
print(index.getCollectionStatistics())

# documents that 
doc_data = pd.DataFrame({'docno': ['b44c3a4829fde82ebbcd30094d2b5b26']})
doc_fetch = pt.text.get_text(index, "text")
print("Raw Contents of docno: b44c3a4829fde82ebbcd30094d2b5b26")
print(doc_fetch.transform(doc_data)['text'].values)
print()

# print indexed terms
print("(term, frequency) for docno: b44c3a4829fde82ebbcd30094d2b5b26")
di = index.getDirectIndex()
doi = index.getDocumentIndex()
lex = index.getLexicon()
docid = 1553
for posting in di.getPostings(doi.getDocumentEntry(docid)):
    termid = posting.getId()
    lee = lex.getLexiconEntry(termid)
    print(f"<{lee.getKey()}, {posting.getFrequency()}>", end=", ")
print()

# print document frequency of term
print("(document, frequency) for term: 'underperform'")
meta = index.getMetaIndex()
inv = index.getInvertedIndex()
le = lex.getLexiconEntry("underperform")
for posting in inv.getPostings(le):
    docno = meta.getItem("docno", posting.getId())
    print(f"<{docno}, {posting.getFrequency()}>", end=", ")
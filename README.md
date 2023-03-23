# Multi-stage Retrieval using SPLADE and T5

An end-to-end Search Engine that can index documents for two-stage retrieval. The system focuses on a multi-stage retrieval architecture with query expansion using SPLADE and BM25 for retrieval, and the T5 text-to-text transformer for re-ranking. The proposed framework was evaluated on the [Complex Document and Entity Collection (CODEC)](https://github.com/grill-lab/CODEC), which consists of a corpus of social science domains across History, Economics and Politics. CODEC also defines a document ranking and an entity ranking task which align with each other to improve document ranking through entity query expansion and topic modelling.

## Getting Started

- Fork (Optional) and clone the repository.

```bash
git clone --recurse-submodules https://github.com/<username>/multi-stage-retrieval-using-splade-and-t5
```

- Initialise a virtual environment (e.g. venv) and install pre-requisites.

```bash
# create a new env (from the repo root)
python3 -m venv venv

# activate env for unix/linux
source venv/bin/activate    

# activate env for windows
./source/Scripts/activate

# install pre-requisites
pip install -r requirements.txt
```

## Try the Experiments

- Make sure you download the whole corpus and save it as `CODEC/corpus/codec_documents.jsonl`.

- Index the corpus by using the following command.

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input CODEC/corpus/codec_documents.jsonl \
  --index index/CODEC \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

- Run the retrieval benchmark experiment

```bash
python experiments/retrieval_ini.py
```

## Start the API

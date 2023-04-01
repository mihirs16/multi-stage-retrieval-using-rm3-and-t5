# Multi-stage Retrieval using SPLADE and T5

An end-to-end Search Engine that can index documents for two-stage retrieval. The system focuses on a multi-stage retrieval architecture with query expansion using SPLADE or RM3 and BM25 for retrieval, and the T5 text-to-text transformer for re-ranking. The proposed framework was evaluated on the [Complex Document and Entity Collection (CODEC)](https://github.com/grill-lab/CODEC), which consists of a corpus of social science domains across History, Economics and Politics. CODEC also defines a document ranking and an entity ranking task which align with each other to improve document ranking through entity query expansion and topic modelling.

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

## Try the Search Engine

- Make sure you download the whole corpus and save it as `CODEC/corpus/codec_documents.jsonl`.

- Start the API Server (Will automatically build the index, or load pre-built index).

```bash
python app.py
```

- Open the `localhost:8000` URL in the browser. Select and Model you want to use and type the query in the search box.

## Try the Experiments

- Make sure you download the whole corpus and save it as `CODEC/corpus/codec_documents.jsonl`.

- Run the following script to show all experiment results and save results to `./results/metrics.csv`.

```bash
python eval.py
```

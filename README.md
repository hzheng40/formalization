# formalization
Final project of CIS 620: Text Formalization with Gradient-free Optimization

## Dependencies

### Stanford CoreNLP

CoreNLP will be used to extract features of text, install CoreNLP following the instructions [here](https://stanfordnlp.github.io/CoreNLP/download.html#steps-to-setup-from-the-official-release).

Before running any python scripts, start the CoreNLP Server by running:

```bash
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref -status_port 9000 -port 9000 -timeout 15000
```

### PyTorch

PyTorch 1.8.1 and torchtext 0.9.1 is used in this project. Install via:

```bash
$ pip3 install torch==1.8.1 torchtext==0.9.1
```

### CMA-ES

Nevergrad 0.4.2 and ray 1.1.0 is used for the gradient free optimization. Install the dependencies via:

```bash
$ pip3 install nevergrad ray
```

### Utilities
nltk, for interface to CoreNLP and other utilities

```bash
$ pip3 install nltk

```
textblob, for sentiment module

```bash
$ pip3 install -U textblob
```

gensim, for w2v

```bash
$ pip3 install gensim
```

numpy, scipy, and scikitlearn

```bash
$ pip3 install numpy scipy scikit-learn
```

tqdm, for showing progress bars
```bash
pip3 install tqdm
```
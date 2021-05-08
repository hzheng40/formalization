# Text Formalization with Gradient Free Optimization in Latent Space
Final project of CIS 620: Text Formalization with Gradient Free Optimization in Latent Space
Authors: Hongrui Zheng, Kailing Zheng

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

## GYAFC Corpus
Please follow the instructions in [https://github.com/raosudha89/GYAFC-corpus](https://github.com/raosudha89/GYAFC-corpus) to request access to the GYAFC-corpus. After extracting the files, create a ```.data``` directory and put the corpus in as ```.data/GYAFC_Corpus```.

## Large models

Download the models from [https://drive.google.com/drive/folders/1tfEzpF3y3ZIlMiB6UnnXSSE3YrrcV2Vc?usp=sharing](https://drive.google.com/drive/folders/1tfEzpF3y3ZIlMiB6UnnXSSE3YrrcV2Vc?usp=sharing) and put both of the files in the directory ```data/saved_models```.

## Experiments

### VAE
To run the experiments involving the VAE with the pre-trained model, run:
```bash
$ python3 vae_train.py --to_train false
```
To retrain the model, you can run the above line without the ```to_train``` argument. You can check the progress of training with tensorboard in another terminal by running
```bash
$ tensorboard --logdir data/tensorboard_log/vae_model_gyafc_weightfix3_nodropout_25000crossover_long_0.0005k/
```

### Formality predictor
To run examples with the formality predictor, run:
```bash
$ python3 evals.py
```
### Gradient free optimization
To run the search, run:
```bash
$ python3 search.py
```

### Text style transfer
To test the full framework of formalization, run:
```bash
$ python3 test_transfer.py
```
The output of the transfer will be at ```data/test_output.ref0```. As a reference, the input of the transfer will be at ```.data/GYAFC_Corpus/Family_Relationships/test/informal.ref0```
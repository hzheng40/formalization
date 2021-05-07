import torch
from torchtext.datasets import PennTreebank, WikiText103, WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf):
    """
    Convert raw datasets to iterators, and full vocab of dataset
    """
    # get vocab
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    for line_test in test_iter:
        counter.update(tokenizer(line_test))
    for line_valid in valid_iter:
        counter.update(tokenizer(line_valid))
    vocab = Vocab(counter,
                  # min_freq=10,
                  max_size=conf.n_vocab - 4,
                  specials=(conf.unk_token,
                            conf.pad_token,
                            conf.start_token,
                            conf.end_token),
                  vectors=conf.vocab_vector)

    # transform func from text to ind
    text_transform = lambda x: [vocab[conf.start_token]] + [vocab[token] for token in tokenizer(x)] + [vocab[conf.end_token]]

    # collate func for dataloader
    def collate_batch(batch):
        text_list = []
        for text in batch:
            processed_text = torch.tensor(text_transform(text))
            text_list.append(processed_text)

        # padding is 3 because index of pad special token is index 3 in vocab
        return pad_sequence(text_list, padding_value=3.0)

    # get batch iterator
    train_dataloader = DataLoader(list(train_iter_copy), batch_size=conf.batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter_copy), batch_size=conf.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(list(valid_iter_copy), batch_size=conf.batch_size, shuffle=True, collate_fn=collate_batch)
    return train_dataloader, test_dataloader, valid_dataloader, vocab

def load_dataset_single(orig_iter, conf, vocab):
    # only the iterators, vocab based on vocab used for training
    tokenizer = get_tokenizer('basic_english')
    # transform func from text to ind
    text_transform = lambda x: [vocab[conf.start_token]] + [vocab[token] for token in tokenizer(x)] + [vocab[conf.end_token]]

    # collate func for dataloader
    def collate_batch(batch):
        text_list = []
        for text in batch:
            processed_text = torch.tensor(text_transform(text))
            text_list.append(processed_text)

        # padding is 3 because index of pad special token is index 3 in vocab
        return pad_sequence(text_list, padding_value=3.0)

    # get batch iterator
    dataloader = DataLoader(list(orig_iter), batch_size=conf.cma_batch_size, shuffle=True, collate_fn=collate_batch)

    return dataloader

def str_to_tensor(s, vocab, conf):
    # tokenizer
    tokenizer = get_tokenizer('basic_english')
    # transform func from text to ind
    text_transform = lambda x: [vocab[conf.start_token]] + [vocab[token] for token in tokenizer(x)] + [vocab[conf.end_token]]
    tensor = torch.tensor(text_transform(s))
    return tensor

def get_ptb(conf):
    """
    Return PennTreeBank iterators
    """
    # raw data
    train_iter, test_iter, valid_iter = PennTreebank(split=('train', 'test', 'valid'))
    train_iter_copy, test_iter_copy, valid_iter_copy = PennTreebank(split=('train', 'test', 'valid'))
    # loader
    train, test, valid, vocab = load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf)
    return train, test, valid, vocab

def get_wiki103(conf):
    """
    Return WikiText 103 iterators
    """
    # raw data
    train_iter, test_iter, valid_iter = WikiText103(split=('train', 'test', 'valid'))
    train_iter_copy, test_iter_copy, valid_iter_copy = WikiText103(split=('train', 'test', 'valid'))
    # loader
    train, test, valid, vocab = load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf)
    return train, test, valid, vocab

def get_wiki2(conf):
    """
    Return WikiText 2 iterators
    """
    # raw data
    train_iter, test_iter, valid_iter = WikiText2(split=('train', 'test', 'valid'))
    train_iter_copy, test_iter_copy, valid_iter_copy = WikiText2(split=('train', 'test', 'valid'))
    # loader
    train, test, valid, vocab = load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf)
    return train, test, valid, vocab

def get_gyafc(conf):
    """
    Return GYAFC iterators
    """
    # train
    with open('.data/GYAFC_Corpus/Family_Relationships/train/formal') as f:
        train_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Family_Relationships/train/informal') as f:
        train_iter = train_iter + f.readlines()

    # test
    with open('.data/GYAFC_Corpus/Family_Relationships/test/formal') as f:
        test_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Family_Relationships/test/informal') as f:
        test_iter = test_iter + f.readlines()

    # valid
    with open('.data/GYAFC_Corpus/Family_Relationships/tune/formal') as f:
        valid_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Family_Relationships/tune/informal') as f:
        valid_iter = valid_iter + f.readlines()

    import copy
    train_iter_copy = copy.deepcopy(train_iter)
    test_iter_copy = copy.deepcopy(test_iter)
    valid_iter_copy = copy.deepcopy(valid_iter)

    # loader
    train, test, valid, vocab = load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf)
    return train, test, valid, vocab

def get_formality_set(conf, vocab):
    """
    Return formality iterators for CMA
    Uses test set
    """

    with open('.data/GYAFC_Corpus/Family_Relationships/test/informal') as f:
        test_iter = f.readlines()

    # loader
    test = load_dataset_single(test_iter, conf, vocab)
    return test

def get_informal_test_set(conf, vocab):
    """
    Return small informal set for testing
    """

    with open('.data/GYAFC_Corpus/Family_Relationships/test/informal.ref0') as f:
        test_iter = f.readlines()

    # loader
    test = load_dataset_single(test_iter, conf, vocab)
    return test

def get_gyafc_music(conf):
    """
    Return GYAFC iterators
    """
    # train
    with open('.data/GYAFC_Corpus/Entertainment_Music/train/formal') as f:
        train_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Entertainment_Music/train/informal') as f:
        train_iter = train_iter + f.readlines()

    # test
    with open('.data/GYAFC_Corpus/Entertainment_Music/test/formal') as f:
        test_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Entertainment_Music/test/informal') as f:
        test_iter = test_iter + f.readlines()

    # valid
    with open('.data/GYAFC_Corpus/Entertainment_Music/tune/formal') as f:
        valid_iter = f.readlines()
    with open('.data/GYAFC_Corpus/Entertainment_Music/tune/informal') as f:
        valid_iter = valid_iter + f.readlines()

    import copy
    train_iter_copy = copy.deepcopy(train_iter)
    test_iter_copy = copy.deepcopy(test_iter)
    valid_iter_copy = copy.deepcopy(valid_iter)

    # loader
    train, test, valid, vocab = load_dataset(train_iter, test_iter, valid_iter, train_iter_copy, test_iter_copy, valid_iter_copy, conf)
    return train, test, valid, vocab

if __name__ == '__main__':
    # simple test
    import yaml
    from argparse import Namespace
    with open('configs/default.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    train, test, valid, vocab = get_gyafc(conf)
import torch
from torchtext.datasets import PennTreebank
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
                  min_freq=10,
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


if __name__ == '__main__':
    # simple test
    import yaml
    from argparse import Namespace
    with open('configs/default.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    train, test, valid, vocab = get_ptb(conf)
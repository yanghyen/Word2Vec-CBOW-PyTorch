from collections import Counter
import random

import torch
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.corpus import brown
import requests, zipfile, io

def subsample_text(text, t=1e-5):
    counter = Counter(text)
    total_count = len(text)
    freqs = {word: count / total_count for word, count in counter.items()}

    subsampled = []
    for word in text:
        f = freqs[word]
        p_drop = 1 - (t / f) ** 0.5
        if random.random() > p_drop:
            subsampled.append(word)
    return subsampled

def load_corpus(name="brown"):
    """
    name: "brown" | "text8"
    """
    if name == "brown":
        nltk.download('brown')
        sentences = brown.sents()
        text = [w.lower() for sent in sentences for w in sent]
    
    elif name == "text8":
        nltk.download('punkt')
        url =  "http://mattmahoney.net/dc/text8.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        text = z.read("text8").decode("utf-8")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return text

def build_vocab(text):
    if isinstance(text, str):
        tokens = text.lower().split()
    else:
        tokens = text
    vocab = set(tokens)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return vocab, word2idx, idx2word

def generate_context_target(text, window_size=2):
    data = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j] for j in range(window_size, 0, -1)] + \
                    [text[i + j] for j in range(1, window_size + 1)]
        target = text[i]
        data.append((context, target))
    return data

class CBOWDataset(Dataset):
    def __init__(self,data, word2idx):
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idx = torch.tensor([self.word2idx[word] for word in context], dtype=torch.long)
        target_idx = torch.tensor(self.word2idx[target], dtype=torch.long)
        return context_idx, target_idx
    
def get_dataloader(config, word2idx):
    text = load_corpus(config.get("dataset", "brown"))
    data = generate_context_target(text, window_size=config["window_size"])
    dataset = CBOWDataset(data, word2idx)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    return dataloader
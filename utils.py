from collections import Counter
from pathlib import Path
import re
import os
import numpy as np
from bs4 import BeautifulSoup
from urlextract import URLExtract

import torch
from torch.utils.data import Dataset

import nltk

nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

# SETUP CONSTANTS
SEED = 42
RAW_DATA_DIR = Path("data/raw")
PREPROCESSED_DATA_DIR = Path("data/preprocessed")

extractor = URLExtract()


def word_tokenize(text):
    return nltk.tokenize.word_tokenize(text)


def clean_html(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def remove_urls(text):
    urls = extractor.find_urls(text)
    for u in urls:
        text = text.replace(u, " URLTOKEN ")
    return text, len(urls), urls


def preprocess_text(text):
    # collapse whitespace, lowercase
    text = re.sub(r"\s+", " ", str(text))
    text = text.strip()
    text = text.lower()
    return text


def tokenize(text):
    if not isinstance(text, str):
        return []
    return nltk.word_tokenize(text)


# Small keyword feature helper
KEYWORD_PATTERN = re.compile(
    r"\b(password|verify|account|bank|login|confirm|click|urgent|reset)\b"
)

# Vocabulary builder


def build_vocab(token_lists, max_vocab, min_freq=2):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab_tokens = ["<PAD>", "<UNK>"] + [
        tok for tok, c in counter.most_common(max_vocab) if c >= min_freq
    ]
    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos


# Preprocess functions
def full_preprocess_text(text, return_urls=False):
    text = clean_html(text)
    text = preprocess_text(text)
    text, n_urls, urls = remove_urls(text)
    if return_urls:
        return text, n_urls, urls
    return text


def preprocess_row(text):
    import pandas as pd

    text, n_urls, urls = full_preprocess_text(text, return_urls=True)
    return pd.Series({"text": text, "n_urls": n_urls, "urls": urls})


def compute_numeric_features(row):
    import pandas as pd

    body = str(row["text"])
    n_upper = sum(1 for c in body if c.isupper())
    n_exclaim = body.count("!")
    n_special = sum(1 for c in body if not c.isalnum() and not c.isspace())
    length = len(body.split())
    has_login_words = int(bool(KEYWORD_PATTERN.search(body)))
    return pd.Series(
        {
            "n_upper": n_upper,
            "n_exclaim": n_exclaim,
            "n_special": n_special,
            "length": length,
            "has_login_words": has_login_words,
        }
    )


class EmailDataset(Dataset):
    def __init__(self, seq_ids, num_feats, labels):
        self.seq_ids = seq_ids
        self.num_feats = num_feats
        self.labels = labels

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(self.seq_ids[idx], dtype=torch.long),
            "num": torch.tensor(self.num_feats[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    @staticmethod
    def collate_batch(batch):
        seqs = [item["seq"] for item in batch]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        maxlen = max(lengths).item()
        padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
        nums = torch.stack([item["num"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"seq": padded, "lengths": lengths, "num": nums, "label": labels}

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
nltk.download('punkt_tab')
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
    return word_tokenize(text)


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

class EmailDataset(Dataset):
    def __init__(self, df, stoi, max_len, scaler=None):
        self.df = df.reset_index(drop=True)
        self.stoi = stoi
        self.max_len = max_len
        self.scaler = scaler
        self.url_extractor = URLExtract()

        # Precompute numeric features
        self.num_feats = []
        for _, row in self.df.iterrows():
            body = row.get("text", "") or ""
            raw = body
            text = clean_html(raw)
            text = preprocess_text(text)
            text, n_urls, urls = remove_urls(text)

            n_upper = sum(
                1 for c in (body) if c.isupper()
            )  # number of uppercase letters
            n_exclaim = (body).count("!")  # number of exclamation marks
            n_special = sum(
                1 for c in (body) if not c.isalnum() and not c.isspace()
            )  # number of special characters
            length = len(text.split())
            has_login_words = int(
                bool(KEYWORD_PATTERN.search(text))
            )  # 1 if any keyword found else 0
            features = [
                n_urls,
                n_upper,
                n_exclaim,
                n_special,
                length,
                has_login_words,
            ]  # feature list

            self.num_feats.append(features)

        self.num_feats = np.array(self.num_feats, dtype=np.float32)
        if scaler is not None:
            self.num_feats = scaler.transform(self.num_feats)

        # Tokenize to ids
        self.seq_ids = []
        for _, row in self.df.iterrows():
            body = row.get("text", "") or ""
            text = clean_html(body)
            text = preprocess_text(text)
            text, _, _ = remove_urls(text)
            toks = tokenize(text)
            ids = [self.stoi.get(tok, self.stoi.get("<UNK>")) for tok in toks][
                : self.max_len
            ]
            self.seq_ids.append(ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.seq_ids[idx]
        num = self.num_feats[idx].astype(np.float32)
        label = int(self.df.loc[idx, "label"])
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "num": torch.tensor(num, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

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

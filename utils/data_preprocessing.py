import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch


stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()
patterns = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"-]+"

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    return ' '.join(tokens)


def process_data(data):
    data['combined_text'] = data['title'] + " " + data['description']
    data['lemmatized_text'] = data['combined_text'].apply(lemmatize)

    model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
    data['combined_vector'] = data['lemmatized_text'].apply(lambda l: model.encode(l, convert_to_tensor=True).cpu().numpy())


# Определение Dataset
class TagDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = torch.tensor(self.encodings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item, label

    def __len__(self):
        return len(self.labels)
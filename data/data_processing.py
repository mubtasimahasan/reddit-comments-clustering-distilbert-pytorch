import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import torch
import re

from utils.config import CFG

def get_data(train_size=0.65, valid_size=0.15):
    dataset = pd.read_csv('data/reddit_data.csv').dropna()
    if CFG.debug:
        dataset = dataset.sample(frac=0.10)
    
    documents = dataset['text'].tolist()
    mapping = {'pcmasterrace': 0, 'news': 1, 'relationships': 2, 'nfl': 3, 'movies': 4}
    labels = dataset['topic'].map(mapping).tolist()
    CFG.num_labels = dataset["topic"].nunique()
    
    # Split the dataset into train, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(documents, labels, test_size=1-train_size, random_state=42)
    valid_texts, test_texts, valid_labels, test_labels = train_test_split(test_texts, test_labels, test_size=valid_size/(1-train_size), random_state=42)
    return (train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels), dataset.topic.unique()


def preprocess_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')    # add/remove regex as required
    SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # unnecessary symbols
    NUMBERS = re.compile('\d+') # numbers
    STOPWORDS = set(stopwords.words('english')) # stopwords
    lemmatizer = WordNetLemmatizer()  # lemmatizer

    # normalize and clean
    text = text.lower() # lowering the text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # removing punc
    text = SYMBOLS_RE.sub('', text) # removing unnecessary symbols
    text = NUMBERS.sub('', text) # removing stop words

    # remove stopwords
    tokens = [word for word in text.split() if word not in STOPWORDS]
    
    # lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)


def tokenize_data(tokenizer, texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=CFG.max_length)
    return encodings

def create_dataset(encodings, labels):
    class RedditDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    return RedditDataset(encodings, labels)

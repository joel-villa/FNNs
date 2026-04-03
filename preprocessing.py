"""
A file for preprocessing IMDB data

The vast majority of the code below comes from the Machine Learning with PyTorch
and Scikit-Learn Textbook
"""
import pyprind
import pandas as pd
import os
import sys
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def load_data_from_aclImdb():
    """
    from text book, w/ a few changes to compile w/ most recent version of pandas
    """
    # change the 'basepath' to the directory of the
    # unzipped movie dataset
    basepath = 'aclImdb'

    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)
    df = pd.DataFrame()

    data = []

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                data.append([txt, labels[l]])
                pbar.update()

    df = pd.DataFrame(data, columns=['review', 'sentiment'])

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('movie_data.csv', index=False, encoding='utf-8')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


df = pd.read_csv('data/movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})
df.head(3)

#Clean the data
df['review'] = df['review'].apply(preprocessor)

#Transform reviews to term frequency inverse document frequency (tf-idf)
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)

count = CountVectorizer()

X = tfidf.fit_transform(count.fit_transform(df['review']))

#Split data
X_df = pd.DataFrame(X.toarray())
X_df['sentiment'] = df['sentiment'].values

split_idx = int(0.7 * len(X_df))

train_df = X_df.iloc[:split_idx]
test_df = X_df.iloc[split_idx:]

train_df.to_csv('data/train_data.csv', index=False, encoding='utf-8')
test_df.to_csv('data/test_data.csv', index=False, encoding='utf-8')


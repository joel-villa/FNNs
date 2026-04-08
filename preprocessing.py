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
    df.to_csv('data/movie_data.csv', index=False, encoding='utf-8')

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


df = pd.read_csv('data/movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})

#Clean the data
df['review'] = df['review'].apply(preprocessor)

#Transform reviews to term frequency inverse document frequency (tf-idf)
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)

# set max features to 5000
count = CountVectorizer(max_features=8000)

x_counts = count.fit_transform(df['review'])
x = tfidf.fit_transform(x_counts)

x_dense = x.toarray().astype(np.float32)

x_df = pd.DataFrame(x_dense)
x_df['sentiment'] = df['sentiment'].values

# Shuffle the data (for randomized reproducible split)
x_df_shuffled = x_df.sample(frac=1, random_state=42).reset_index(drop=True)

# split data here
split_idx = int(0.7 * len(x_df))
train_df = x_df.iloc[:split_idx]
test_df = x_df.iloc[split_idx:]

# train_df.to_csv('data/train_data.csv', index=False, encoding='utf-8')
# test_df.to_csv('data/test_data.csv', index=False, encoding='utf-8')

train_x = train_df.drop(columns=['sentiment']).to_numpy(dtype=np.float32)
train_y = train_df['sentiment'].to_numpy()

test_x = test_df.drop(columns=['sentiment']).to_numpy(dtype=np.float32)
test_y = test_df['sentiment'].to_numpy()

np.savez_compressed(
    'data/imdb_tfidf_data8000.npz',
    x_train=train_x,
    y_train=train_y,
    x_test=test_x,
    y_test=test_y
)

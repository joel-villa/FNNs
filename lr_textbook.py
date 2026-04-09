from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
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

# tfidf = TfidfVectorizer(strip_accents=None,
#                         lowercase=False,
#                         preprocessor=None)


def tokenizer(text):
    return text.split()

# tokenizer('runners like running and thus they run')
# ['runners', 'like', 'running', 'and', 'thus', 'they', 'run']

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# tokenizer_porter('runners like running and thus they run')
# ['runner', 'like', 'run', 'and', 'thu', 'they', 'run']

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf':[False],
        'vect__norm':[None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]
lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5,
                           verbose=2, n_jobs=1)

import time

start = time.perf_counter() #Timing training
gs_lr_tfidf.fit(X_train, y_train)
end = time.perf_counter()

print(f"Training time: {time}")


print(f'CV Accuracy: {gs_lr_tfidf.best_score_}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test)}')
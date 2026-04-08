"""
For loading in the reviews and sentiments from csv files, or from npz files
"""
import pandas as pd
import numpy as np

def csv_load():
    df_test = pd.read_csv('data/test.csv', encoding='utf-8')
    return df_test

def npz_load(size=None):
    if size == 8000:
        data = np.load('data/imdb_tfidf_data8000.npz')
    elif size == 10000:
        data = np.load('data/imdb_tfidf_data10000.npz')
    elif size == 15000:
        data = np.load('data/imdb_tfidf_data15000.npz')
    elif size == 20000:
        data = np.load('data/imdb_tfidf_data20000.npz')
    else:
        data = np.load('data/imdb_tfidf_data.npz')


    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test
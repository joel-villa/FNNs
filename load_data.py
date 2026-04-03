"""
For loading in the reviews and sentiments from csv files, or from npz files
"""
import pandas as pd
import numpy as np

def csv_load():
    df_test = pd.read_csv('data/test.csv', encoding='utf-8')
    return df_test

def npz_load():
    data = np.load('data/imdb_tfidf_data.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test
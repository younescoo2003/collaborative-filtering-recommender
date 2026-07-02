import pandas as pd
from surprise import Dataset, Reader

def load_movielens():
    data = Dataset.load_builtin('ml-100k')
    return data

def load_movielens_df():
    data = Dataset.load_builtin('ml-100k')
    df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])
    return df

def load_sample_data(filepath='data/sample.dat'):
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(filepath, reader=reader)
    return data
from collections import namedtuple
import numpy as np
import pandas as pd


Data = namedtuple('Data', ['size', 'clicked_words', 'clicked_entities', 'words', 'entities', 'labels'])


def load_data(args):
    train_df = read(args.train_file, split_words=args.split_words)
    test_df = read(args.test_file, split_words=args.split_words)
    uid2words, uid2entities = aggregate(train_df, args.max_click_history)
    train_data = transform(train_df, uid2words, uid2entities)
    test_data = transform(test_df, uid2words, uid2entities)
    return train_data, test_data


def read(file, split_words=True):
    df = pd.read_table(file, sep='\t', header=None, names=['user_id', 'words', 'entities', 'label'])
    df['entities'] = df['entities'].map(lambda x: [int(i) for i in x.split(',')])
    if split_words:
        df['words'] = df['words'].map(lambda x: [int(i) for i in x.split(',')])
    return df


def aggregate(train_df, max_click_history):
    uid2words = dict()
    uid2entities = dict()
    pos_df = train_df[train_df['label'] == 1]
    for user_id in set(pos_df['user_id']):
        df_user = pos_df[pos_df['user_id'] == user_id]
        words = np.array(df_user['words'].tolist())
        entities = np.array(df_user['entities'].tolist())
        indices = np.random.choice(list(range(0, df_user.shape[0])), size=max_click_history, replace=True)
        uid2words[user_id] = words[indices]
        uid2entities[user_id] = entities[indices]
    return uid2words, uid2entities


def transform(df, uid2words, uid2entities):
    df['clicked_words'] = df['user_id'].map(lambda x: uid2words[x])
    df['clicked_entities'] = df['user_id'].map(lambda x: uid2entities[x])
    data = Data(size=df.shape[0],
                clicked_words=np.array(df['clicked_words'].tolist()),
                clicked_entities=np.array(df['clicked_entities'].tolist()),
                words=np.array(df['words'].tolist()),
                entities=np.array(df['entities'].tolist()),
                labels=np.array(df['label']))
    return data

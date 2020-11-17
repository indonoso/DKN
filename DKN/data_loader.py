from collections import namedtuple
import numpy as np
import pandas as pd


Data = namedtuple('Data', ['size', 'clicked_words', 'clicked_entities', 'words', 'entities', 'labels'])


def load_data(args):
    train_df = read(args.train_file, args.max_title_length, args.max_title_length, split_words=args.split_words)
    test_df = read(args.test_file, args.max_title_length, args.max_title_length, split_words=args.split_words)
    train_df_clicked = aggregate(train_df, args.max_click_history)
    test_df_clicked = aggregate(test_df, args.max_click_history)
    train_data = transform(train_df, train_df_clicked)
    test_data = transform(test_df, test_df_clicked)
    return train_data, test_data


def read(file, max_words, max_entities, split_words=True):
    df = pd.read_table(file, sep='\t', header=None, names=['user_id', 'words', 'entities', 'label'])
    df['entities'] = df['entities'].map(lambda x: __pad_truncate([int(i) for i in x.split(',')],
                                                                 max_entities))
    if split_words:
        df['words'] = df['words'].map(lambda x: __pad_truncate([int(i) for i in x.split(',')],
                                                               max_words))
    return df


def __pad_truncate(x, max_length):
    if len(x) < max_length:
        return x + [0] * (max_length - len(x))
    else:
        return x[:max_length]


def aggregate(train_df, max_click_history):
    pos_df = train_df[train_df['label'] == 1]
    index_col = ('clicked_words', 'clicked_entities')

    def agg(df_user):
        words = np.array(df_user['words'].tolist())
        entities = np.array(df_user['entities'].tolist())
        indices = np.random.choice(list(range(0, df_user.shape[0])), size=max_click_history, replace=True)
        return pd.Series((words[indices], entities[indices]), index=index_col)
    r = pos_df.groupby(['user_id'], as_index=False).apply(agg)
    return r


def transform(df, clicked):
    df = pd.merge(df, clicked[['clicked_words', 'user_id']], on='user_id')
    df = pd.merge(df, clicked[['clicked_entities', 'user_id']], on='user_id')
    data = Data(size=df.shape[0],
                clicked_words=np.array(df['clicked_words'].tolist()),
                clicked_entities=np.array(df['clicked_entities'].tolist()),
                words=np.array(df['words'].tolist()),
                entities=np.array(df['entities'].tolist()),
                labels=np.array(df['label']))
    return data

from collections import namedtuple
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

Data = namedtuple('Data', ['size', 'clicked_words', 'clicked_entities', 'words', 'entities', 'labels'])


class DataLoader:
    def __init__(self, file_path=None, use_bert_embeddings=True, max_text_length=10, max_click_history=10, **kwargs):
        self.max_text_length = max_text_length
        self.split_words = not use_bert_embeddings
        self.max_click_history = max_click_history
        self.file_path = file_path
        self._load_data()

    def _load_data(self):
        logger.debug('Reading files')
        df = self.read()
        logger.debug('Aggregating columns')
        df_clicked = self.aggregate(df)
        logger.debug('Transforming data')
        self.data = self.transform(df, df_clicked)
        return self.data

    def read(self):
        df = pd.read_table(self.file_path, sep='\t', header=None, names=['user_id', 'words', 'words_encoded', 'entities', 'label'])
        df['entities'] = df['entities'].map(lambda x: self.__pad_truncate([int(i) for i in x.split(',')],
                                                                     self.max_text_length))
        if self.split_words:
            df['words'] = df['words_encoded'].map(lambda x: self.__pad_truncate([int(i) for i in x.split(',')],
                                                                   self.max_text_length))
        del df['words_encoded']
        return df

    @staticmethod
    def __pad_truncate(x, max_length):
        if len(x) < max_length:
            return x + [0] * (max_length - len(x))
        else:
            return x[:max_length]

    def aggregate(self, df):
        pos_df = df[df['label'] == 1]
        index_col = ('clicked_words', 'clicked_entities')

        def agg(df_user):
            words = np.array(df_user['words'].tolist())
            entities = np.array(df_user['entities'].tolist())
            indices = np.random.choice(list(range(0, df_user.shape[0])), size=self.max_click_history, replace=True)
            return pd.Series((words[indices], entities[indices]), index=index_col)
        r = pos_df.groupby(['user_id'], as_index=False).apply(agg)
        return r

    @staticmethod
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

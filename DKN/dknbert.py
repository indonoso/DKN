import numpy as np
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
from .pipeline import CachedFeatureExtractionPipeline
from .base_dkn import DKN
import pickle


class DKNBert(DKN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.scibert = CachedFeatureExtractionPipeline(self.word_dim, self.max_text_length, 'scibert', model, tokenizer,
                                                       task='word_embeddings')
        self.model_params['scibert'] = self.scibert

    def _build_inputs(self):
        with tf.compat.v1.name_scope('input'):
            self.clicked_words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, self.max_click_history, self.max_text_length, self.word_dim],
                name='clicked_words')
            self.clicked_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_click_history, self.max_text_length], name='clicked_entities')
            self.words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, self.max_text_length, self.word_dim], name='words')
            self.entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_text_length], name='entities')
            self.labels = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self):
        with tf.compat.v1.name_scope('embedding'):
            self.entity_embeddings = tf.Variable(np.load(self.entity_embeddings_path), dtype=np.float32, name='entity')
            self.params.append(self.entity_embeddings)

            if self.use_context:
                context_embs = np.load(self.context_embeddings_path)
                self.context_embeddings = tf.Variable(context_embs, dtype=np.float32, name='context')
                self.params.append(self.context_embeddings)

            if self.transform:
                self.entity_embeddings = tf.compat.v1.layers.dense(
                    self.entity_embeddings, units=self.entity_dim, activation=tf.nn.tanh, name='transformed_entity',
                    kernel_regularizer=tf.keras.regularizers.l2(0.5 * self.l2_weight))
                if self.use_context:
                    self.context_embeddings = tf.compat.v1.layers.dense(
                        self.context_embeddings, units=self.entity_dim, activation=tf.nn.tanh,
                        name='transformed_context', kernel_regularizer=tf.keras.regularizers.l2(0.5 * self.l2_weight))

        user_embeddings, item_embeddings = self._attention()
        self.scores_unnormalized = tf.reduce_sum(input_tensor=user_embeddings * item_embeddings, axis=1)
        self.scores = tf.sigmoid(self.scores_unnormalized, name='scores')

    def _prepare_data_attention(self):
        clicked_words = tf.reshape(self.clicked_words, shape=[-1, self.max_text_length, self.word_dim])
        clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, self.max_text_length])
        return clicked_words, clicked_entities

    def _prepare_data_kcnn(self, words, entities):
        embedded_entities = tf.nn.embedding_lookup(params=self.entity_embeddings, ids=entities)
        return words, embedded_entities

    @staticmethod
    def transform_feed_dict(data, start, end, model):
        return [np.array([model.scibert(cw.tolist()) for cw in data.clicked_words[start:end]]),
                data.clicked_entities[start:end],
                np.array(model.scibert(data.words[start:end].tolist())),
                data.entities[start:end],
                data.labels[start:end]]

    def save_prediction_model(self):
        self.save_session()
        self.scibert.save_cache()
        with open(str(self.output_path) + ".pickle", 'wb') as f:
            pickle.dump({'params': self.model_params,
                         'transform_feed_dict': self.transform_feed_dict}, f)
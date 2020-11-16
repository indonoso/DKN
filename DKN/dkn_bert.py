import numpy as np
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
from .pipeline import CachedFeatureExtractionPipeline
from .base_dkn import DKN


class DKN_Bert(DKN):
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.scibert = CachedFeatureExtractionPipeline(args.word_dim, args.max_title_length, model, tokenizer, task='word_embeddings')
        super().__init__(args)

    def _build_inputs(self, args):
        with tf.compat.v1.name_scope('input'):
            self.clicked_words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_click_history, args.max_title_length, args.word_dim],
                name='clicked_words')
            self.clicked_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_click_history, args.max_title_length], name='clicked_entities')
            self.words = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, args.max_title_length, args.word_dim], name='words')
            self.entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, args.max_title_length], name='entities')
            self.labels = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, args):
        with tf.compat.v1.name_scope('embedding'):
            self.entity_embeddings = tf.Variable(np.load(args.entity_embeddings), dtype=np.float32, name='entity')
            self.params.append(self.entity_embeddings)

            if args.use_context:
                context_embs = np.load(args.context_embeddings)
                self.context_embeddings = tf.Variable(context_embs, dtype=np.float32, name='context')
                self.params.append(self.context_embeddings)

            if args.transform:
                self.entity_embeddings = tf.compat.v1.layers.dense(
                    self.entity_embeddings, units=args.entity_dim, activation=tf.nn.tanh, name='transformed_entity',
                    kernel_regularizer=tf.keras.regularizers.l2(0.5 * (args.l2_weight)))
                if args.use_context:
                    self.context_embeddings = tf.compat.v1.layers.dense(
                        self.context_embeddings, units=args.entity_dim, activation=tf.nn.tanh,
                        name='transformed_context', kernel_regularizer=tf.keras.regularizers.l2(0.5 * (args.l2_weight)))

        user_embeddings, item_embeddings = self._attention(args)
        self.scores_unnormalized = tf.reduce_sum(input_tensor=user_embeddings * item_embeddings, axis=1)
        self.scores = tf.sigmoid(self.scores_unnormalized)

    def _prepare_data_attention(self, args):
        clicked_words = tf.reshape(self.clicked_words, shape=[-1, args.max_title_length, args.word_dim])
        clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, args.max_title_length])
        return clicked_words, clicked_entities

    def _prepare_data_kcnn(self, words, entities):
        embedded_entities = tf.nn.embedding_lookup(params=self.entity_embeddings, ids=entities)
        return words, embedded_entities

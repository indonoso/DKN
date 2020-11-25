import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class DKN:
    def __init__(self, transform=False, use_bert_embeddings=None, word_embeddings_path=None,
                 entity_embeddings_path=None, context_embeddings_path=None, use_context=False, max_click_history=10,
                 max_text_length=10, entity_dim=32, word_dim=32, l2_weight=0.01, filter_sizes=(1, 2), n_filters=128,
                 lr=0.001, batch_size=128, n_epochs=10, output_path=None):
        self.model_params = dict(transform=transform, use_bert_embeddings=use_bert_embeddings,
                                 word_embeddings_path=word_embeddings_path,
                                 entity_embeddings_path=entity_embeddings_path,
                                 context_embeddings_path=context_embeddings_path, use_context=use_context,
                                 max_click_history=max_click_history, max_text_length=max_text_length,
                                 entity_dim=entity_dim,
                                 word_dim=word_dim, l2_weight=l2_weight, filter_sizes=filter_sizes, n_filters=n_filters,
                                 lr=lr, batch_size=batch_size, n_epochs=n_epochs, output_path=output_path)
        self.output_path = output_path
        if output_path is None:
            raise Warning("The output path has not been set. The model will not be saved.")
        # Embeddings
        self.transform = transform

        # Word Embeddings
        self.use_bert_embeddings = use_bert_embeddings
        self.word_embeddings_path = word_embeddings_path
        if not(self.use_bert_embeddings or self.word_embeddings_path):
            raise ValueError('Neither bert embeddings or Word2Vec has been set')

        # Entity
        self.entity_embeddings_path = entity_embeddings_path

        # Context
        self.context_embeddings_path = context_embeddings_path
        self.use_context = use_context

        # Tensor size
        self.max_click_history = max_click_history
        self.max_text_length = max_text_length
        self.entity_dim = entity_dim
        self.word_dim = word_dim

        # Model
        self.l2_weight = l2_weight
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.lr = lr

        # Training

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.params = []  # for computing regularization loss
        self._build_inputs()
        self._build_model()
        self._build_train()
        self.session = None

    def _prepare_data_attention(self):
        clicked_words = tf.reshape(self.clicked_words, shape=[-1, self.max_text_length])
        clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, self.max_text_length])
        return clicked_words, clicked_entities

    def _prepare_data_kcnn(self, words, entities):
        embedded_words = tf.nn.embedding_lookup(params=self.word_embeddings, ids=words)
        embedded_entities = tf.nn.embedding_lookup(params=self.entity_embeddings, ids=entities)
        return embedded_words, embedded_entities

    def _build_inputs(self):
        with tf.compat.v1.name_scope('input'):
            self.clicked_words = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_click_history, self.max_text_length], name='clicked_words')
            self.clicked_entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_click_history, self.max_text_length], name='clicked_entities')
            self.words = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_text_length], name='words')
            self.entities = tf.compat.v1.placeholder(
                dtype=tf.int32, shape=[None, self.max_text_length], name='entities')
            self.labels = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self):
        with tf.compat.v1.name_scope('embedding'):
            self.word_embeddings = tf.Variable(np.load(self.word_embeddings_path), dtype=np.float32, name='word')
            self.entity_embeddings = tf.Variable(np.load(self.entity_embeddings_path), dtype=np.float32, name='entity')
            self.params.append(self.word_embeddings)
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
        self.scores = tf.sigmoid(self.scores_unnormalized)

    def _attention(self):
        # (batch_size * max_click_history, max_title_length)
        clicked_words, clicked_entities = self._prepare_data_attention()

        with tf.compat.v1.variable_scope('kcnn', reuse=tf.compat.v1.AUTO_REUSE):  # reuse the variables of KCNN
            # (batch_size * max_click_history, title_embedding_length)
            # title_embedding_length = n_filters_for_each_size * n_filter_sizes
            clicked_embeddings = self._kcnn(clicked_words, clicked_entities)

            # (batch_size, title_embedding_length)
            item_embeddings = self._kcnn(self.words, self.entities)

        # (batch_size, max_click_history, title_embedding_length)
        clicked_embeddings = tf.reshape(
            clicked_embeddings, shape=[-1, self.max_click_history, self.n_filters * len(self.filter_sizes)])

        # (batch_size, 1, title_embedding_length)
        item_embeddings_expanded = tf.expand_dims(item_embeddings, 1)

        # (batch_size, max_click_history)
        attention_weights = tf.reduce_sum(input_tensor=clicked_embeddings * item_embeddings_expanded, axis=-1)

        # (batch_size, max_click_history)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        # (batch_size, max_click_history, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # (batch_size, title_embedding_length)
        user_embeddings = tf.reduce_sum(input_tensor=clicked_embeddings * attention_weights_expanded, axis=1)

        return user_embeddings, item_embeddings

    def _kcnn(self, words, entities):
        # (batch_size * max_click_history, max_title_length, word_dim) for users
        # (batch_size, max_title_length, word_dim) for items
        embedded_words, embedded_entities = self._prepare_data_kcnn(words, entities)

        # (batch_size * max_click_history, max_title_length, full_dim) for users
        # (batch_size, max_title_length, full_dim) for items
        if self.use_context:
            embedded_contexts = tf.nn.embedding_lookup(params=self.context_embeddings, ids=entities)
            concat_input = tf.concat([embedded_words, embedded_entities, embedded_contexts], axis=-1)
            full_dim = self.word_dim + self.entity_dim * 2
        else:
            concat_input = tf.concat([embedded_words, embedded_entities], axis=-1)
            full_dim = self.word_dim + self.entity_dim

        # (batch_size * max_click_history, max_title_length, full_dim, 1) for users
        # (batch_size, max_title_length, full_dim, 1) for items
        concat_input = tf.expand_dims(concat_input, -1)

        outputs = []
        for filter_size in self.filter_sizes:
            filter_shape = [filter_size, full_dim, 1, self.n_filters]
            w = tf.compat.v1.get_variable(name='w_' + str(filter_size), shape=filter_shape, dtype=tf.float32)
            b = tf.compat.v1.get_variable(name='b_' + str(filter_size), shape=[self.n_filters], dtype=tf.float32)
            if w not in self.params:
                self.params.append(w)

            # (batch_size * max_click_history, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for users
            # (batch_size, max_title_length - filter_size + 1, 1, n_filters_for_each_size) for items
            conv = tf.nn.conv2d(input=concat_input, filters=w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # (batch_size * max_click_history, 1, 1, n_filters_for_each_size) for users
            # (batch_size, 1, 1, n_filters_for_each_size) for items
            pool = tf.nn.max_pool2d(input=relu, ksize=[1, self.max_text_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID', name='pool')
            outputs.append(pool)

        # (batch_size * max_click_history, 1, 1, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, 1, 1, n_filters_for_each_size * n_filter_sizes) for items
        output = tf.concat(outputs, axis=-1)

        # (batch_size * max_click_history, n_filters_for_each_size * n_filter_sizes) for users
        # (batch_size, n_filters_for_each_size * n_filter_sizes) for items
        output = tf.reshape(output, [-1, self.n_filters * len(self.filter_sizes)])

        return output

    def _build_train(self):
        with tf.compat.v1.name_scope('train'):
            self.base_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores_unnormalized))
            self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                self.l2_loss = tf.add(self.l2_loss, self.l2_weight * tf.nn.l2_loss(param))
            if self.transform:
                self.l2_loss = tf.add(self.l2_loss, tf.compat.v1.losses.get_regularization_loss())
            self.loss = self.base_loss + self.l2_loss
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, data, start, end):
        return sess.run(self.optimizer, self.get_feed_dict(data, start, end))

    def eval(self, labels, scores):
        auc = roc_auc_score(y_true=labels, y_score=scores)
        return auc

    def get_labels_scores(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores], feed_dict)
        return labels, scores

    def get_feed_dict(self, data, start, end):
        feed_dict = {self.clicked_words: data.clicked_words[start:end],
                     self.clicked_entities: data.clicked_entities[start:end],
                     self.words: data.words[start:end],
                     self.entities: data.entities[start:end],
                     self.labels: data.labels[start:end]}
        return feed_dict

    def save_session(self):
        if self.output_path:
            saver = tf.compat.v1.train.Saver()
            save_path = saver.save(self.session, self.output_path)
            print("Model saved in path: %s" % save_path)
        else:
            raise ValueError('Output path is None')

    def __getstate__(self):
        self.save_session()
        return self.model_params

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_params = state
        self.session = tf.compat.v1.saved_model.loader.load(self.output_path)

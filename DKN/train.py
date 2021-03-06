from .dkn_sentence_embedding import DKNSentenceEmbedding
from .base_dkn import DKN
import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
tf.compat.v1.disable_eager_execution()
logger = logging.getLogger(__name__)


def train(train_data, test_data, val_data=None, n_epochs=1, batch_size=128, fast_train=True, predictions=False, **kwargs):
    tf.compat.v1.reset_default_graph()
    if kwargs.get('use_bert_embeddings'):
        model = DKNSentenceEmbedding(n_epochs=n_epochs, batch_size=batch_size, **kwargs)
        logger.debug('Using Bert Embeddings')
    else:
        model = DKN(n_epochs=n_epochs, batch_size=batch_size, **kwargs)
        logger.debug('Using W2V embeddings')

    with tf.compat.v1.Session() as sess:
        model.session = sess
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        train_auc, test_auc, model_predictions = None, None, None
        for step in tqdm(range(n_epochs), desc='Epochs'):
            logger.debug('Starting training')
            start_list = list(range(0, train_data.size, batch_size))
            np.random.shuffle(start_list)
            for start in tqdm(start_list, desc='Batches', mininterval=1, position=0, leave=True):
                end = start + batch_size
                model.train(sess, train_data, start, end)
            if predictions and fast_train and (step == n_epochs - 1):
                logger.info('Making predictions')
                model_predictions = (get_prediction(val_data, batch_size, model),
                                     get_prediction(test_data, batch_size, model))

            elif (fast_train and step == n_epochs - 1) or (not fast_train):
                logger.info('Evaluation - training')
                train_auc = evaluation(train_data, batch_size, model)
                logger.info('Evaluation - validation ')
                test_auc = evaluation(test_data, batch_size, model)
                logger.info('epoch %d  train_auc %.4f  test_auc: %.4f' % (step, train_auc, test_auc))

        if model.output_path:
            model.save_prediction_model()
    return train_auc, test_auc, model_predictions


def evaluation(data, batch_size, model):
    labels, scores = [], []
    for start in range(0, data.size, batch_size):
        l, s = model.predict(data, start, start + batch_size)
        labels.append(l)
        scores.append(s)
    return roc_auc_score(np.hstack(labels), np.hstack(scores))


def get_prediction(data, batch_size, model):
    labels, scores = [], []
    for start in range(0, data.size, batch_size):
        l, s = model.predict(data, start, start + batch_size)
        labels.append(l)
        scores.append(s)
    return np.hstack(labels), np.hstack(scores)

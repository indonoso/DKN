from .dknbert import DKNBert
from .base_dkn import DKN
import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()
logging.basicConfig(level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p',
                    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
import pickle
logger = logging.getLogger(__name__)


def train(train_data, test_data, n_epochs=1, batch_size=128, output_path=None, **kwargs):
    tf.compat.v1.reset_default_graph()
    if kwargs.get('use_bert_embeddings'):
        model = DKNBert(output_path=output_path, **kwargs)
        logger.debug('Using Bert Embeddings')
    else:
        print(kwargs)
        model = DKN(output_path=output_path, **kwargs)
        logger.debug('Using W2V embeddings')

    with tf.compat.v1.Session() as sess:
        model.session = sess
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        for step in tqdm(range(n_epochs), desc='Epochs'):
            logger.debug('Starting training')
            # training
            start_list = list(range(0, train_data.size, batch_size))
            np.random.shuffle(start_list)
            for start in tqdm(start_list, desc='Batches', mininterval=3, position=0, leave=True):
                end = start + batch_size
                model.train(sess, train_data, start, end)

            logger.info('Evaluation - training')
            train_auc = evaluation(train_data, batch_size, model)
            logger.info('Evaluation - validation ')
            test_auc = evaluation(test_data, batch_size, model)
            print('epoch %d  train_auc %.4f  test_auc: %.4f' % (step, train_auc, test_auc))

        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)


def evaluation(data, batch_size, model):
    labels, scores = [], []
    for start in range(0, data.size, batch_size):
        l, s = model.get_labels_scores(model.get_feed_dict(data, start, start + batch_size))
        labels.append(l)
        scores.append(s)
    return model.eval(np.hstack(labels), np.hstack(scores))

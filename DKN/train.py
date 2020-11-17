from .dkn_bert import DKN_Bert
from .base_dkn import DKN
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG,datefmt='%d/%m/%Y %I:%M:%S %p',
                    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def get_feed_dict_words(model, data, start, end):
    feed_dict = {model.clicked_words: np.array([model.scibert(cw.tolist()) for cw in data.clicked_words[start:end]]),
                 model.clicked_entities: data.clicked_entities[start:end],
                 model.words: np.array(model.scibert(data.words[start:end].tolist())),
                 model.entities: data.entities[start:end],
                 model.labels: data.labels[start:end]}

    return feed_dict


def get_feed_dict_word_ids(model, data, start, end):
    feed_dict = {model.clicked_words: data.clicked_words[start:end],
                 model.clicked_entities: data.clicked_entities[start:end],
                 model.words: data.words[start:end],
                 model.entities: data.entities[start:end],
                 model.labels: data.labels[start:end]}
    return feed_dict


def train(args, train_data, test_data):

    if args.user_bert_embeddings:
        model = DKN_Bert(args)
        get_feed_dict = get_feed_dict_words
        logger.debug('Using Bert Embeddings')
    else:
        model = DKN(args)
        get_feed_dict = get_feed_dict_word_ids
        logger.debug('Using W2V embeddings')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        for step in tqdm(range(args.n_epochs), desc='Epochs'):
            logger.debug('Starting training')
            # training
            start_list = list(range(0, train_data.size, args.batch_size))
            np.random.shuffle(start_list)
            for start in tqdm(start_list, desc='Batches'):
                end = start + args.batch_size
                model.train(sess, get_feed_dict(model, train_data, start, end))
            # evaluation
            # train_auc = model.eval(sess, get_feed_dict(model, train_data, 0, train_data.size))
            test_auc = model.eval(sess, get_feed_dict(model, test_data, 0, test_data.size))
            print('epoch %d   test_auc: %.4f' % (step, test_auc))

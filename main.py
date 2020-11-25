import argparse
from DKN import DataLoader
from DKN import train_dkn
from DKN.base_dkn import DKNPredict
from DKN.train import evaluation

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='../data/news/train.txt', help='path to the training file')
parser.add_argument('--test_file', type=str, default='../data/news/test.txt', help='path to the test file')
parser.add_argument('--output_path', type=str, default='model', help='path to the test file')

parser.add_argument('--entity_embeddings_path', type=str, default='', help='path to the training file')
parser.add_argument('--word_embeddings_path', type=str, default='', help='path to the test file')
parser.add_argument('--context_embeddings_path', type=str, default='', help='path to the test file')

parser.add_argument('--transform', type=str2bool, default=True, help='whether to transform entity embeddings')
parser.add_argument('--use_context', type=str2bool, default=False, help='whether to use context embeddings')
parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')
parser.add_argument('--n_filters', type=int, default=128, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--entity_dim', type=int, default=50,
                    help='dimension of entity embeddings, please ensure that the specified input file exists')
parser.add_argument('--word_dim', type=int,
                    help='dimension of word embeddings, please ensure that the specified input file exists')
parser.add_argument('--max_text_length', type=int, default=10,
                    help='maximum length of news titles, should be in accordance with the input datasets')

parser.add_argument('--use_bert_embeddings', type=str2bool, default=False,
                    help='use Bert to get word embeddings. Requires split_words to be False')

kwargs = vars(parser.parse_args())

train_data = DataLoader(file_path=kwargs.pop('train_file'), **kwargs)
test_data = DataLoader(file_path=kwargs.pop('test_file'), **kwargs)
# train_dkn(train_data.data, test_data.data, **kwargs)

model = DKNPredict.load_prediction_model('model')
print(evaluation(test_data.data, kwargs['batch_size'], model))
model.session.close()

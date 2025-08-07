import os
from os.path import join
from pathlib import Path
import sys
sys.path.append('..')
cur_dir = Path(os.getcwd())
par_dir = cur_dir.parent.absolute()
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append(str(par_dir))
from scGCN.src.utils import *
from tensorflow.python.saved_model import tag_constants
from scGCN.src.models import scGCN
from utilsForCompeting import evaluator
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
tf.set_random_seed(seed)

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"

    input_dir = "Please enter input directory path"
    output_dir = "Please enter output directory path"
    

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('f', '', 'kernel')  # to run in jupyter kernels
    flags.DEFINE_string('dataset', join(input_dir, exp_id), 'data dir')
    flags.DEFINE_string('output', join(output_dir, f'{exp_id}_results'), 'predicted results')
    flags.DEFINE_bool('graph', True, 'select the optional graph.')
    flags.DEFINE_string('model', 'scGCN','Model string.')
    flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0,
                       'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10,
                         'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    print(FLAGS.dataset)
    # Load data
    adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label, index_guide = load_data(
        FLAGS.dataset, exp_id, rgraph=FLAGS.graph)

    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = scGCN

    # Define placeholders
    placeholders = {
        'support':
        [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features':
        tf.sparse_placeholder(tf.float32,
                              shape=tf.constant(features[2], dtype=tf.int64)),
        'labels':
        tf.placeholder(tf.float32, shape=(None, labels_binary_train.shape[1])),
        'labels_mask':
        tf.placeholder(tf.int32),
        'dropout':
        tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
        tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                            placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1]

    # Initialize session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    cost_val = []
    for epoch in range(FLAGS.epochs):
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=construct_feed_dict(
            features, support, labels_binary_train, train_mask, placeholders))
        
        # Validation
        cost, acc = evaluate(features, support, labels_binary_val, val_mask, placeholders)
        cost_val.append(cost)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:04d} train_loss= {outs[1]:.5f} train_acc= {outs[2]:.5f} val_loss= {cost:.5f} val_acc= {acc:.5f}")

    # Test
    test_cost, test_acc = evaluate(features, support, labels_binary_test, test_mask, placeholders)
    print(f"Test set results: cost= {test_cost:.5f} accuracy= {test_acc:.5f}")

    # Prediction
    all_mask = np.array([True] * len(train_mask))
    labels_binary_all = new_label
    
    feed_dict_all = construct_feed_dict(features, support, labels_binary_all,
                                        all_mask, placeholders)
    feed_dict_all.update({placeholders['dropout']: FLAGS.dropout})

    activation_output = sess.run(model.activations, feed_dict=feed_dict_all)[1]
    predict_output = sess.run(model.outputs, feed_dict=feed_dict_all)

    ab = sess.run(tf.nn.softmax(predict_output))
    all_prediction = sess.run(
        tf.equal(sess.run(tf.argmax(ab, 1)),
                 sess.run(tf.argmax(labels_binary_all.astype("int32"), 1))))

    acc_pred = np.sum(all_prediction[pred_mask]) / np.sum(pred_mask)
    print(f'Prediction accuracy: {acc_pred:.4f}')

    # Evaluation
    src_label = pd.read_csv(join(FLAGS.dataset, 'Label1.csv'))['type'].values
    tgt_label = pd.read_csv(join(FLAGS.dataset, 'Label2.csv'))['type'].values
    tgt_shr_mask = np.in1d(tgt_label, np.unique(src_label))
    all_pr = np.argmax(ab, axis=1)
    all_gt = np.argmax(labels_binary_all.A, axis=1)
    
    kn_data_pr = all_pr[pred_mask][tgt_shr_mask]
    kn_data_gt = all_gt[pred_mask][tgt_shr_mask]
    
    closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)

if __name__ == "__main__":
    main()

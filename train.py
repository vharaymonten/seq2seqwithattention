import argparse
import tensorflow as tf 
import numpy as np
from helper import data_pipeline, tokenize, batch, save_hparams
import json 
import os 
from seq2seq import Model

parser = argparse.ArgumentParser()

parser.add_argument("-v-","--verbose", nargs="?", type=bool, const=1, help="enable verbose mode (optional)")
parser.add_argument("-ckpt_path", "--ckpt_path",nargs="?", help="path to save model",type=str)
parser.add_argument("-src_dataset", "--src_dataset", help="path to source dataset file", type=str)
parser.add_argument("-tgt_dataset", "--tgt_dataset", help="path to target dataset file", type=str)
parser.add_argument('-hparams', "--hparams", help="path to hyperparameters file (.json)", type=str)
parser.add_argument("-epoch", "--epoch", const=1, nargs="?", help="number of epoch", type=int)
parser.add_argument("-n_step", "--n_step",nargs="?", const=100, help="print every n step ", type=int)
parser.add_argument("-test_dir", "--test_dir", help="Just specify the directory of test data if train\
 dataset has been specified\n (both test data filenames must be same as train dataset) ", type=str)
parser.add_argument("--test_data_split", "-test_data_split", help='test data split (0,1 - 0.9) ', type=float)
args = parser.parse_args()

json_path = args.hparams
fp = open(json_path, 'r')
hparams = json.load(fp)


epoch = args.epoch 
print_n_step = args.n_step
verbose = args.verbose
ckpt_path = args.ckpt_path
tgt_sentences, tgt_metadata = data_pipeline(args.tgt_dataset)
src_sentences, src_metadata = data_pipeline(args.src_dataset)
tgt_max_len = tgt_metadata.max_time_step
src_max_len = src_metadata.max_time_step

src_inputs = np.array([tokenize(sentence,src_max_len, src_metadata, source=True, reverse=True)\
                          for sentence in src_sentences])
tgt_outputs = np.array([tokenize(sentence,tgt_max_len, tgt_metadata, source=False, reverse=False)\
                          for sentence in tgt_sentences])


hparams['tgt_vocab_size'] = tgt_metadata.vocab_size
hparams['src_vocab_size'] = src_metadata.vocab_size
hparams['dec_max_time_step'] = tgt_max_len

train_graph = tf.Graph()
eval_graph = tf.Graph()

#train_sess = tf.Session(graph=train_graph)
#eval_sess = tf.Session(graph=eval_graph)


with train_graph.as_default():
    train_model = Model("train", hparams)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with eval_graph.as_default():
    eval_model = Model("eval", hparams)



with tf.Session(graph=train_graph) as sess:
    sess.run(init)
    for e in range(1, epoch+1):
        for step, (X, y) in enumerate(batch(hparams['batch_size'], src_inputs[128:], tgt_outputs[128:])):
            
            loss = train_model.train_batch(sess, X, y)
            if verbose:
                step += 1 
                if step % 100 == 0: 
                    print(" Epoch : {} Step : {} Loss : {} ".format(epoch, step, loss))

    train_model.saver.save(sess, ckpt_path)

with tf.Session(graph=eval_graph) as sess:
    sess.run(tf.global_variables_initializer())
    eval_model.saver.restore(sess, ckpt_path)
    loss = eval_model.eval(sess, src_inputs[:128],tgt_outputs[:128])
    print(loss)

# Save new hparams for later use in inference mode
save_hparams(hparams)
import argparse
import tensorflow as tf 
import numpy as np
from helper import data_pipeline, tokenize, batch, save_hparams, tokens2sentence, save_metadata, load_metadata
import json 
import os 
from seq2seq import Model

parser = argparse.ArgumentParser()

parser.add_argument("-v-","--verbose", nargs="?", type=bool, const=1, help="enable verbose mode (optional)")
parser.add_argument("-ckpt_prefix", "--ckpt_prefix",nargs="?", help="checkpoint path prefix",type=str)
parser.add_argument("-src_dataset", "--src_dataset", help="path to source dataset file", type=str)
parser.add_argument("-tgt_dataset", "--tgt_dataset", help="path to target dataset file", type=str)
parser.add_argument('-hparams', "--hparams", help="path to hyperparameters file (.json)", type=str)
parser.add_argument("-epoch", "--epoch", const=1, nargs="?", help="number of epoch", type=int)
parser.add_argument("-print_nsteps", "--print_nsteps",nargs="?", const=100, help="print every n step ", type=int)
parser.add_argument("-test_dir", "--test_dir", help="Just specify the directory of test data if train\
 dataset has been specified\n (both test data filenames must be same as train dataset) ", type=str)
parser.add_argument("--test_data_split", "-test_data_split", help='test data split (0,1 - 0.9) ', type=float)
#parser.add_argument("-metadata_path", "--metadata_path",  help='directory for metadata to be saved')

args = parser.parse_args()

json_path = args.hparams
fp = open(json_path, 'r')
hparams = json.load(fp)


epoch = args.epoch 
print_nsteps = args.print_nsteps
verbose = args.verbose
ckpt_prefix = args.ckpt_prefix

batch_size = hparams['batch_size']

if os.path.isdir(ckpt_prefix) == False:
    os.mkdir(ckpt_prefix)

ckpt_path = os.path.join(ckpt_prefix, 'model.ckpt')


tgt_sentences, tgt_metadata = data_pipeline(args.tgt_dataset, padding=True)
src_sentences, src_metadata = data_pipeline(args.src_dataset, padding=True)

src_inputs = np.array([tokenize(sentence, src_metadata, source=True, reverse=True)\
                          for sentence in src_sentences])
tgt_outputs = np.array([tokenize(sentence, tgt_metadata, source=False, reverse=False)\
                          for sentence in tgt_sentences])



save_metadata(tgt_metadata, "tgt_metadata.dill")
save_metadata(src_metadata, "src_metadata.dill")


hparams['tgt_vocab_size'] = tgt_metadata.vocab_size
hparams['src_vocab_size'] = src_metadata.vocab_size
hparams['dec_max_time_step'] = tgt_metadata.max_time_step
save_hparams(json_path, hparams)


train_graph = tf.Graph()
eval_graph = tf.Graph()


train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)


with train_graph.as_default():
    train_model = Model("train", hparams)
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
with eval_graph.as_default():
    eval_model = Model("eval", hparams)
    eval_init = tf.global_variables_initializer()
with infer_graph.as_default():
    infer_model = Model("infer", hparams)
    infer_init = tf.global_variables_initializer()



for e in range(1, epoch+1):
    if e == 1:
        train_sess.run(init)
        eval_sess.run(eval_init)
    iterator = batch(batch_size, src_inputs[batch_size:], tgt_outputs[batch_size:])
    train_model.train(train_sess, iterator, print_nsteps, e=e)
    train_model.saver.save(train_sess, ckpt_path)

    #######################
    eval_model.saver.restore(eval_sess, ckpt_path)
    eval_loss = eval_model.eval(eval_sess, src_inputs[:batch_size], tgt_outputs[:batch_size])
    print("Epoch {} Eval loss {}".format(e, eval_loss))
'''
with tf.Session(graph=train_graph) as sess:
    sess.run(init)
    for e in range(1, epoch+1):
        for step, (X, y) in enumerate(batch(hparams['batch_size'], src_inputs[128:], tgt_outputs[128:])):
            
            loss = train_model.train_batch(sess, X, y)
            if verbose:
                step += 1 
                if step % 100 == 0: 
                    print(" Epoch : {} Step : {} Loss : {} ".format(epoch, step, loss))

    saver.save(sess, ckpt_path)

with tf.Session(graph=eval_graph) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    loss = eval_model.eval(sess, src_inputs[:128], tgt_outputs[:128])
    print(loss)
'''
with tf.Session(graph=infer_graph) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    output = infer_model.inference(sess, src_inputs[0].reshape(1,-1))
    print(tokens2sentence(output[0], tgt_metadata))

# Save new hparams for later use in inference mode
train_sess.close()
eval_sess.close()
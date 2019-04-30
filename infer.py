import tensorflow as tf
from seq2seq import Model
from helper import data_pipeline, tokenize, sentence2tokens, load_metadata, tokens2sentence
import argparse, json, os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("-ckpt_prefix", "--ckpt_prefix",nargs="?", help="checkpoint path prefix",type=str)
parser.add_argument('-hparams', "--hparams", help="path to hyperparameters file (.json)", type=str)


args = parser.parse_args()

tgt_sentences, tgt_metadata = data_pipeline('dataset/french.txt')
src_sentences, src_metadata = data_pipeline('dataset/english.txt')

src_metadata = load_metadata('src_metadata.dill')
tgt_metadata = load_metadata('tgt_metadata.dill')

src_inputs = np.array([tokenize(sentence, src_metadata, source=True, reverse=True)\
                          for sentence in src_sentences])

fp = open(args.hparams, 'r')
hparams = json.load(fp)
ckpt_path = os.path.join(args.ckpt_prefix, 'model.ckpt')


infer_graph  = tf.Graph()

with infer_graph.as_default():
    infer_model = Model("infer", hparams)
    
with tf.Session(graph=infer_graph) as sess:
    sess.run(tf.global_variables_initializer())
    infer_model.saver.restore(sess, ckpt_path)
    while True:
        sentence = input("please input an english sentence...\n")
        tokens = sentence2tokens(sentence, src_metadata)
        output = infer_model.inference(sess, np.array(tokens).reshape(1,-1))
        print(tokens2sentence(output[0], tgt_metadata))

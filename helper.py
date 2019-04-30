import numpy as np
import re 
import dill
from collections import Counter, namedtuple
from unicodedata import normalize
import string
import os 
import json 

GO_ID  = 0
EOS_ID = 1
UNK_ID = 2 
PAD_ID = 3

dataset_dir = 'dataset/'
metadata_dir = 'metadata'
ckpt_path = 'checkpoints'


 
def open_file(path_to_file,encoding='utf-8'):
    with open(path_to_file,'r',encoding='utf-8') as f:
        text = f.read()
        #close the file object
        f.close()
    return text

def to_sentences(text):
    #remove white space and down casing all words 
    text = text.lower()
    text = text.strip()
    sentences = text.split('\n')
    return sentences 


def sentence_length(sentences):
    min_len = min([len(s.split()) for s in sentences])
    max_len = max([len(s.split()) for s in sentences])
    
    return max_len, min_len

def to_vocab(sentences):
    # create frequency table for all words  
    vocab = Counter()
    
    for s in sentences:
        vocab.update(s.split())
    return vocab

def create_lookup_table(vocab, padding=True):
    if padding:
        tokens = dict(GO_ID=GO_ID,EOS_ID=EOS_ID, UNK_ID=UNK_ID, PAD_ID=PAD_ID)
    else:
         tokens = dict(GO_ID=GO_ID,EOS_ID=EOS_ID, UNK_ID=UNK_ID)
    word2int = {word:idx+len(tokens) for idx,word in enumerate(vocab)}
    word2int.update(tokens)    
    int2word = {idx:word for word, idx in word2int.items()}
    return word2int, int2word 


def tokenize(sentence, metadata ,source=True, reverse=False):    
    tokens = [metadata.word2int[word] for word in sentence.split()]
    s_len = len(tokens)
    max_sentence_length = metadata.max_time_step
    if source:
        pads = np.ones(max_sentence_length) * PAD_ID
        pads[:s_len] = tokens
    else:
        #add one extra space for EOS_ID
        pads = np.ones(max_sentence_length+1) * PAD_ID
        pads[-1] = 1 
        pads[:s_len] = tokens
        
    if reverse:
        return list(reversed(pads))
    else:
        return pads
   
    
def min_occurance(vocab, min_occurance=2):
    new_vocab = set()
    temp = {word for word, c in vocab.items() if c >= min_occurance}
    new_vocab.update(temp)
    return new_vocab 
 
def in_vocab(sentence, vocab):
    new_sentence = []
    for word in sentence.split() :
            if word in vocab:
                new_sentence.append(word)
            else:
                new_sentence.append('UNK_ID')
    new_sentence = ' '.join(new_sentence)
    return new_sentence
def update_dataset(sentences, vocab):
    updated_dataset = list()
    for sentence in sentences:
        new_sentence = in_vocab(sentence, vocab)
        updated_dataset.append(new_sentence)
    
    return updated_dataset 


def clean_sentence(sentence):
    #remove non-printable charaters 
    sentence = re.sub('[^%s]' % string.printable, '', sentence)
    #remove non-printable charaters 
    sentence = re.sub('[%s]' % string.punctuation, '', sentence)
    sentence = normalize('NFD', sentence).encode('ascii','ignore')
    sentence = sentence.decode('UTF-8')
    sentence = sentence.strip()
    return sentence
def clean_sentences(sentences):
    cleaned = list()
    for s in sentences:
            s = clean_sentence(s)
            cleaned.append(s)
    return cleaned

def tokens2sentence(tokens, metadata):
    words = [metadata.int2word[token] for token in tokens]
    sentence = ' '.join(words)
    return sentence

def sentence2tokens(sentence, metadata):
    sentence = clean_sentence(sentence)
    sentence = in_vocab(sentence, metadata.vocab)
    tokens  = tokenize(sentence, metadata, reverse=True, source=True)
    return np.array(tokens)

def sentences2tokens(sentences, metadata):
    tokens_arr = []
    for sentence in sentences:
        tokens_arr.append(sentences, metadata)
    return tokens_arr

def split_data(x, n_split=0.2):
    split_idx = np.ceil(x.shape[0]*0.2)
    x_train = x[split_idx:]
    x_eval = x[:split_idx]
    return x_train, x_eval

def batch(batch_size, X, Y, time_major=False):
    assert(X.shape[0] == Y.shape[0]), "Dimension mismatch! decoder_inputs.shape != decoder_outputs.shape"

    n = X.shape[0]
    steps = n // batch_size
    
    start = 0
    for step in range(1, steps+1):
        end = batch_size*step
        
        yield X[start:end], Y[start:end]
        start = end 
        
    
def data_pipeline(path_to_file, padding=False):
    doc = open_file(path_to_file)
    sentences = clean_sentences(to_sentences(doc))    
    
    vocab = min_occurance(to_vocab(sentences))
    sentences = update_dataset(sentences, vocab)
    
    word2int, int2word = create_lookup_table(vocab, padding)
    metadata = namedtuple("Metadata",["vocab","word2int", "int2word", "vocab_size", "max_time_step"])
    metadata = metadata(list(word2int.keys()), word2int, int2word, len(word2int.keys()), sentence_length(sentences)[0])
    return sentences, metadata
    
def save_metadata(metadata, fname):
    if os.path.isdir(metadata_dir) == False:
        os.mkdir(metadata_dir)
    fp = os.path.join(metadata_dir, fname)
    fp = open(fp, 'wb')
    #_metadata =[metadata.vocab, metadata.word2int, metadata.int2word, metadata.vocab_size, metadata.max_time_step]
    dill.dump(metadata, fp)
    fp.close()

def load_metadata(fname):
    fp = os.path.join(metadata_dir, fname)
    file = open(fp, 'rb')
    metadata = dill.load(file)
    file.close()
    return metadata


def save_hparams(fp, hparams):
    fp = open(fp,"w")
    json.dump(hparams, fp, indent=4)
    fp.close()

def save_sentences(sentences, fp):
    file = open(fp, 'w')
    doc = '\n'.join(sentences)
    file.write(doc)
    file.close()
    
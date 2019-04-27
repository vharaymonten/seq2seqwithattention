import numpy as np
import re 
import pickle 
from collections import Counter, namedtuple
from unicodedata import normalize
import string
import os 
import json 


dataset_dir = 'dataset/'


def save_hparams(hparams):
    fp = open("hyperparameters.json","w")
    json.dump(hparams, fp)
    fp.close()
 
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

def create_lookup_table(vocab, padding=False):
    
    tokens = dict(GO_ID=0,EOS_ID=1, UNK_ID=2, PAD_ID=3)
            
    word2int = {word:idx+len(tokens) for idx,word in enumerate(vocab)}
    word2int.update(tokens)    
    int2word = {idx:word for word, idx in word2int.items()}
    
        

    return word2int, int2word 


def tokenize(sentence, max_sentence_length, metadata ,source=True, reverse=False):    
    tokens = [metadata.word2int[word] for word in sentence.split()]
    s_len = len(tokens)
    if source:
        pads = np.ones(max_sentence_length) * 3
        pads[:s_len] = tokens
        
    else:

        #add one extra space for EOS_ID
        pads = np.ones(max_sentence_length+1)*3
        pads[-1] = 1
        pads[:s_len] = tokens
        
    if reverse:
        return list(reversed(pads))
    else:
        return pads
   
    
def min_occurance(vocab, min_occurance=4):
    new_vocab = set()
    temp = {word for word, c in vocab.items() if c >= min_occurance}
    new_vocab.update(temp)
    return new_vocab 
 
def update_dataset(sentences, vocab):
    updated_dataset = list()
    for sentence in sentences:
        new_sentence = list()
        for word in sentence.split() :
            if word in vocab:
                new_sentence.append(word)
            else:
                new_sentence.append('UNK_ID')
        new_sentence = ' '.join(new_sentence)
        updated_dataset.append(new_sentence)
    
    return updated_dataset 

def clean_sentences(sentences):
    re_printable = re.compile('[^%s]' % string.printable)
    re_punctuation = re.compile('[%s]' % string.punctuation)
    cleaned = list()
    
    for s in sentences:
            # normalize unicode charaters 
            s = normalize('NFD', s).encode('ascii','ignore')
            s = s.decode('UTF-8')
        
            #remove non-printable charaters 
            s = re_printable.sub('',s)
            #remove punctuation
            s = re_punctuation.sub('', s)
            s = s.strip()
            cleaned.append(s)
    return cleaned

def tokens2sentence(tokens, int2word):
    words = [int2word[token] for token in tokens]
    sentence = ' '.join(words)
    return sentence


def save_sentences(sentences, fp):
    file = open(fp, 'w')
    doc = '\n'.join(sentences)
    file.write(doc)
    file.close()
    

def batch(batch_size, X, Y, shuffle=True, time_major=False):
    assert(X.shape[0] == Y.shape[0]), "Dimension error ! decoder_inputs.shape != decoder_outputs.shape"
    if shuffle:
        np.random.shuffle(X)
        np.random.shuffle(Y)

    n = X.shape[0]
    steps = n // batch_size
    
    start = 0
    for step in range(1, steps+1):
        end = batch_size*step
        
        yield X[start:end], Y[start:end]
        start = end 
        
    
def data_pipeline(path_to_file, ):
    doc = open_file(path_to_file)
    sentences = clean_sentences(to_sentences(doc))    
    
    vocab = min_occurance(to_vocab(sentences))
    sentences = update_dataset(sentences, vocab)
    
    word2int, int2word = create_lookup_table(vocab)
    metadata = namedtuple("Metadata",["vocab","word2int", "int2word", "vocab_size", "max_time_step"])
    metadata = metadata(word2int.keys(), word2int, int2word, len(word2int.items()), sentence_length(sentences)[0])
    return sentences, metadata
    
def save_metadata(metadata, fp):
    fp = open(fp, 'wb')
    pickle.dump(metadata, fp)
    fp.close()


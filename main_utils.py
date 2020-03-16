from sacred import Experiment
import json
import collections
from data import Document, Sentence, Word
from ingredients.corpus import ing as corpus_ingredient, read_jsonl, load_data
import re
import numpy as np
import pickle
import gensim
from model_lstm import Summarization
import time
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import tensorflow as tf


ex = Experiment(name="utils experiment", ingredients=[corpus_ingredient])
@ex.config
def config():
    step = "train"
    path = '/home/minhaj/machine-learning/mysum2/dataset/train.jsonl'
    embedding_path = '/home/minhaj/machine-learning/mysum2/model50/idwiki_word2vec_50.model'
    embedding_size = 50

@ex.capture
def build_dict(step,path):
    if step == 'train':
        wordss = []
        data = read_jsonl(path,remove_puncts=True)
        for docs in data:
            for words in docs.summary.words:
                wordss.append(words.lower())
            for para in docs.paragraphs:
                for sent in para:
                   for words in sent:
                       wordss.append(words)
        word_counter = collections.Counter(wordss).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        data = read_jsonl(path,remove_puncts=True)
        maxdoclen = max([len(docs.words) for docs in data])
        data = read_jsonl(path,remove_puncts=True)
        maxsummarylen = max([len(docs.summary.words) for docs in data])
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)
        with open("vocab/word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)
        with open("vocab/len/article.pickle", "wb") as f:
            pickle.dump(maxdoclen, f)
        with open("vocab/len/summary.pickle", "wb") as f:
            pickle.dump(maxsummarylen, f)
    elif step == 'valid':
        with open("vocab/word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)
        with open("vocab/len/article.pickle", "rb") as f:
            maxdoclen = pickle.load(f)
        with open("vocab/len/summary.pickle", "rb") as f:
            maxsummarylen =  pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return word_dict, reversed_dict, maxdoclen,maxsummarylen

def build_dataset(step, path,word_dict, article_max_len, summary_max_len):
    if step == "train":
        data = read_jsonl(path,remove_puncts=True)
        i = 1
        X = []
        Y = []
        for docs in data:
            X.append(docs.words) 
            Y.append(docs.summary.words) 
        summary2 = []
        content2 = []
        for aa in X:
            content = []
            for aaa in aa:
                content.append(word_dict.get(aaa,1))
            content2.append(content)
        for bb in Y:
            summary = []
            for bbb in bb:
                summary.append(word_dict.get(bbb,1))
            summary2.append(summary)
        xx = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in content2]
        yy = [d + (summary_max_len - len(d)) * [word_dict["<padding>"]] for d in summary2]
        return xx,yy
    elif step == "valid":
        data = read_jsonl(path,remove_puncts=True)
        i = 1
        X = []
        Y = []
        for docs in data:
            X.append(docs.words) 
        content2 = []
        for aa in X:
            content = []
            for aaa in aa:
                content.append(word_dict.get(aaa,1))
            content2.append(content)

        xx = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in content2]
        return xx

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

def get_init_embedding(embedding_path,reversed_dict, embedding_size):
    print("Loading Word2Vec vectors...")
    #word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
    word_vectors = gensim.models.KeyedVectors.load(embedding_path)
     
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors[word]
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)
    return np.array(word_vec_list)

@ex.automain
def my_main(step,path,embedding_path,embedding_size):
    word_dict,reversed_dict,article_max_len,summary_max_len = build_dict('valid',path)
#    X,Y = build_dataset('train', path,word_dict, article_max_len, summary_max_len)
    batch_size=64
    num_epochs=10
    #init_embedding = get_init_embedding(embedding_path,reversed_dict,embedding_size)
    train_x, train_y = build_dataset("train",path, word_dict, article_max_len, summary_max_len)
    batches = batch_iter(train_x, train_y,batch_size, num_epochs)
    #print(init_embedding)
    num_batches_per_epoch = (len(train_x) - 1) // batch_size + 1
    print(num_batches_per_epoch)
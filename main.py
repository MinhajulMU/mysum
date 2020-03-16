from sacred import Experiment
import json
import collections
from data import Document, Sentence, Word
from ingredients.corpus import ing as corpus_ingredient, read_jsonl
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


ex = Experiment(name="main experiment", ingredients=[corpus_ingredient])


@ex.config
def my_config():
    recipients = "swdw"
    path = '/home/minhaj/machine-learning/dataset/cek.jsonl'
    embedding_path = 'model50/idwiki_word2vec_50.model'
    size_layer = 128
    num_layers = 2
    embedding_size = 50
    batch_size = 8
    epoch = 3
    default_path = "/home/minhaj/machine-learning/mysum/"

@ex.capture()
def create_vocab(path):
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
    word_dict["PAD"] = 0
    word_dict["GO"] = 1
    word_dict["EOS"] = 2
    word_dict["UNK"] = 3
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    with open("vocab/word_dict.pickle", "wb") as f:
        pickle.dump(word_dict, f)
    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return word_dict, reversed_dict

def build_dataset(path,dic,UNK=1):
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
            content.append(dic.get(aaa,UNK))
        content2.append(content)
    for bb in Y:
        summary = []
        for bbb in bb:
            summary.append(dic.get(bbb,UNK))
        summary.append(2)
        summary2.append(summary)
    return content2,summary2


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens

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
def my_main(path,embedding_path,embedding_size,default_path):
    if not os.path.exists(default_path + "lstm_seq2seq_model"):
        os.mkdir(default_path + "lstm_seq2seq_model")
        old_model_checkpoint_path = default_path + "lstm_seq2seq_model"
    else:
        old_model_checkpoint_path = default_path + "lstm_seq2seq_model"

    word_dict,reversed_dict = create_vocab(path)
    content,summary = build_dataset(path,word_dict)
    init_embedding = get_init_embedding(embedding_path,reversed_dict,embedding_size)

    size_layer = 128
    num_layers = 2
    batch_size = 8
    epoch = 3
    PAD = 0
    print(init_embedding.shape)
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = Summarization(size_layer, num_layers, embedding_size, reversed_dict,init_embedding)
    sess.run(tf.global_variables_initializer())


    saver = tf.train.Saver(tf.global_variables())
    if 'old_model_checkpoint_path' in globals():
        print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
        saver.restore(sess, old_model_checkpoint_path )

    for EPOCH in range(epoch):
        lasttime = time.time()
        total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0
        train_X, train_Y = shuffle(content, summary)
        pbar = tqdm(range(0, len(train_X), batch_size), desc='train minibatch loop')
        for k in pbar:
            batch_x, _ = pad_sentence_batch(train_X[k: min(k+batch_size,len(train_X))], PAD)
            batch_y, _ = pad_sentence_batch(train_Y[k: min(k+batch_size,len(train_X))], PAD)
    #    print(batch_x)
            
            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer], feed_dict={model.X:batch_x,model.Y:batch_y})   
            
            total_loss += loss
            total_accuracy += acc
            pbar.set_postfix(cost=loss, accuracy = acc)
            
            
        total_loss /= (len(train_X) / batch_size)
        total_accuracy /= (len(train_X) / batch_size)
        
            
        print('epoch: %d, avg loss: %f, avg accuracy: %f'%(EPOCH, total_loss, total_accuracy))
        saver.save(sess, old_model_checkpoint_path)



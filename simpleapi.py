from flask import Flask,jsonify,request
import json
import gensim
import pickle
from nltk import word_tokenize
import string
import nltk
import os
import time
start = time.perf_counter()
from newspaper import  Article
from seq2seq_model import Model
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

default_path = "/home/minhaj/Unduhan/mysum3-20200315T084652Z-001/mysum3/"
num_hidden=150
num_layers=2
beam_width=10
glove="store_true"
embedding_size=50
embedding_path = default_path + 'model50/idwiki_word2vec_50.model'
learning_rate=1e-3
batch_size=64
num_epochs=10
keep_prob = 0.8
toy=False
with_model="store_true"

def build_dict():
    with open(default_path +"vocab/word_dict.pickle", "rb") as f:
        word_dict = pickle.load(f)
    with open(default_path +"vocab/len/article.pickle", "rb") as f:
        maxdoclen = pickle.load(f)
    with open(default_path +"vocab/len/summary.pickle", "rb") as f:
        maxsummarylen =  pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return word_dict, reversed_dict, maxdoclen,maxsummarylen

def replacess(lines):
    filter_word = ['<padding>','<unk>'];
    querywords = lines.split()
    resultwords  = [word for word in querywords if word.lower() not in filter_word]
    result = ' '.join(resultwords)
    return result

def build_dataset2(word_input,word_dict, article_max_len, summary_max_len):
    data = [word_input]
    i = 1
    X = []
    for docs in data:
        X.append(docs) 
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


app = Flask(__name__)


@app.route("/",methods=['GET','POST'])
def haiii():
    if (request.method == 'POST'):
        some_json = request.get_json()
        urls = some_json['url']
        return some_json['url']
    else:
        return jsonify({'about':'hello world'})


@app.route('/predict', methods=['POST'])
def predict():
    some_json = request.get_json()
    urls = some_json['url']
    article = Article(urls)
    article.download()
    article.parse()
    nltk.download('punkt')
    article.nlp()
    punctuationss = '@#$%^&*();:!-_=+[]{\/}|"`<>?'
    table = str.maketrans('UNK', 'UNK', punctuationss)
    words = word_tokenize(article.text)
    stripped = [w.lower().replace(w.lower(),'UNK') if w.lower() in punctuationss else w.lower() for w in words]
    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict()
    valid_x = build_dataset2(stripped, word_dict, article_max_len, summary_max_len)

    tf.reset_default_graph()
    with tf.Session() as sess:
      print("Loading saved model...")
      model = Model(reversed_dict, article_max_len, summary_max_len,True,keep_prob,embedding_size,num_hidden,num_layers,learning_rate,beam_width,glove,embedding_path)
      saver = tf.train.Saver(tf.global_variables())
      ckpt = tf.train.get_checkpoint_state(default_path + "saved_model/")
      saver.restore(sess, ckpt.model_checkpoint_path)        
      batches = batch_iter(valid_x, [0] * len(valid_x), batch_size, 1)
      print("Writing summaries to 'result.txt'...")

      for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]
        valid_feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: list(batch_x),
                    model.X_len: batch_x_len,
                }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]
        summary_array = []
        for line in prediction_output:
          summary = list()
          for word in line:
            if word == "</s>":
              break
            if word not in summary:
              summary.append(word)
              summary_array.append(" ".join(summary))
          
          lines = " ".join(summary)
          result = replacess(lines)
          return result

if __name__ == '__main__':
    app.run(debug=True)


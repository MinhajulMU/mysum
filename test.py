from sacred import Experiment
import time
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from seq2seq_model import Model
from main_utils import build_dict, build_dataset, batch_iter
from ingredients.corpus import ing as corpus_ingredient, read_jsonl, load_data
# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ex = Experiment(name="main experiment", ingredients=[corpus_ingredient])
  
@ex.config
def config():
    default_path = "/home/minhaj/machine-learning/mysum2/"
    path = default_path + 'dataset/dev.02.jsonl'
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


@ex.automain
def main(default_path, path,embedding_path,keep_prob,embedding_size,num_hidden,num_layers,learning_rate,beam_width,glove,batch_size,num_epochs):
    print("Loading dictionary...")
    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid",path)
    print("Loading training dataset...")
    valid_x = build_dataset("valid",path, word_dict, article_max_len, summary_max_len)
    
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


#            print(batch_x.shape)

            valid_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: list(batch_x),
                model.X_len: batch_x_len,
            }

            prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
            prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]
            summary_array = []

            with open(default_path + "result.txt", "a") as f:
                for line in prediction_output:
                    summary = list()
                    for word in line:
                        if word == "</s>":
                            break
                        if word not in summary:
                            summary.append(word)
                    summary_array.append(" ".join(summary))

        print('Summaries have been generated')
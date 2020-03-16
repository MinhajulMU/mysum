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
    path = default_path + 'dataset/train.jsonl'
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

    toy=False #"store_true"

    with_model="store_true"

@ex.automain
def main(default_path, path,embedding_path,keep_prob,embedding_size,num_hidden,num_layers,learning_rate,beam_width,glove,batch_size,num_epochs):
    if not os.path.exists(default_path + "saved_model"):
        os.mkdir(default_path + "saved_model")
    else:
        old_model_checkpoint_path = default_path + 'saved_model'
    print("Building dictionary...")
    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train",path)
    print("Loading training dataset...")
    train_x, train_y = build_dataset("train",path, word_dict, article_max_len, summary_max_len)

    tf.reset_default_graph()

    with tf.Session() as sess:
        model = Model(reversed_dict, article_max_len, summary_max_len+1,False,keep_prob,embedding_size,num_hidden,num_layers,learning_rate,beam_width,glove,embedding_path)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # if 'old_model_checkpoint_path' in globals():
        #     print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
        #     saver.restore(sess, old_model_checkpoint_path )

        batches = batch_iter(train_x, train_y,batch_size, num_epochs)
        num_batches_per_epoch = (len(train_x) - 1) // batch_size + 1

        print("\nIteration starts.")
        print("Number of batches per epoch :", num_batches_per_epoch)
        for batch_x, batch_y in batches:
            print(batch_x.shape)
            # batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            # batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
            # batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
            # batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

            # batch_decoder_input = list(
            #     map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
            # batch_decoder_output = list(
            #     map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

            # train_feed_dict = {
            #     model.batch_size: len(batch_x),
            #     model.X: list(batch_x),
            #     model.X_len: batch_x_len,
            #     model.decoder_input: batch_decoder_input,
            #     model.decoder_len: batch_decoder_len,
            #     model.decoder_target: batch_decoder_output
            # }
            # _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)
            # if step % 1000 == 0:
            #     print("step {0}: loss = {1}".format(step, loss))

            # if step % num_batches_per_epoch == 0:
            #     hours, rem = divmod(time.perf_counter() - start, 3600)
            #     minutes, seconds = divmod(rem, 60)
            #     saver.save(sess, default_path + "saved_model/model.ckpt", global_step=step)
            #     print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
            #     "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")
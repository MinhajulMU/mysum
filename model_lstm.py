import tensorflow as tf



class Summarization:
    def __init__(self, size_layer, num_layers, embedded_size,
                 reversed_dict,init_embedding):
        
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer,initializer=tf.orthogonal_initializer(),reuse=reuse,name="cell_lstm")
        
        GO = 1
        EOS = 3
        self.X = tf.placeholder(tf.int32, [None, None],name="X")
        self.Y = tf.placeholder(tf.int32, [None, None],name="Y")
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        
        init_embeddings = tf.constant(init_embedding, dtype=tf.float32)
        encoder_embedding = tf.get_variable("encoder_embedding", initializer=init_embeddings)
        decoder_embedding = tf.get_variable("decoder_embedding", initializer=init_embeddings)
        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, self.X)
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(cell = encoder_cells, 
                                                                 inputs = encoder_embedded, 
                                                                 sequence_length = self.X_seq_len,
                                                                 dtype = tf.float32)
        
        encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], list(reversed_dict.values()).index('GO')), main], 1)
        dense = tf.layers.Dense(len(reversed_dict)) #50 is sample
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, decoder_input),
                sequence_length = self.Y_seq_len,
                time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = dense)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = decoder_embedding,
                start_tokens = tf.tile(tf.constant([list(reversed_dict.values()).index('GO')], dtype=tf.int32), [batch_size]),
                end_token = list(reversed_dict.values()).index('EOS'))
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = predicting_helper,
                initial_state = encoder_state,
                output_layer = dense)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.X_seq_len))
        self.predicting_ids = predicting_decoder_output.sample_id
        
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost,name="minimizee")
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
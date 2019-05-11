import tensorflow as tf
import numpy as np
from IPython import embed

class LSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0,num_hidden=100):
        print(sequence_length)
        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, 768], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
        
        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss
        
        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #self.W = tf.get_variable(name="W", shape=self.input_x.shape, initializer=tf.constant_initializer(np.array(self.input_x)), trainable=False)
            self.embedded_chars = tf.convert_to_tensor(self.input_x)
            #self.embedded_chars = tf.transpose(self.embedded_chars)
            #self.embedded_chars = tf.expand_dims(self.embedded_chars,-1)
            

            print(self.embedded_chars)
            #self.W = tf.get_variable(name="W", shape=self.input_x.shape, initializer=tf.constant_initializer(self.input_x), trainable=False)
            #self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            #self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #print(self.embedded_chars)
            #print(tf.convert_to_tensor(self.input_x))
       

        #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        
        #2. LSTM LAYER ######################################################################
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
        #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.embedded_chars,dtype=tf.float32)
        #self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.W,dtype=tf.float32)
        #embed()
        
        #val2 = tf.transpose(self.lstm_out, [1, 0, 2])
        val2 = tf.transpose(self.lstm_out, [1, 0, 2])
        print(val2,1234)
        last = tf.gather(val2, int(val2.get_shape()[0])-1)
        print(last)
        
        out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))
        
        with tf.name_scope("output"):
            #lstm_final_output = val[-1]
            #embed()
            self.scores = tf.nn.xw_plus_b(last, out_weight,out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")
        
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")
        
        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")
        """
        with tf.name_scope("precision"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.precision = self.TP / (self.TP + self.FP)

        with tf.name_scope("recall"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.recall = self.TP / (self.TP + self.FN)

        with tf.name_scope("f1"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.precision = self.TP / (self.TP + self.FP)
            self.recall = self.TP / (self.TP + self.FN)
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        """
        print ("LOADED LSTM!")
    


class LSTM_CNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,num_hidden=100):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, 768], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        #self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        #self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            #self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars = tf.convert_to_tensor(self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #2. LSTM LAYER ######################################################################
        self.lstm_cell = tf.contrib.rnn.LSTMCell(768,state_is_tuple=True)
        #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.embedded_chars,dtype=tf.float32)
        #embed()

        self.lstm_out_expanded = tf.expand_dims(self.lstm_out, -1)

        #2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CONVOLUTION LAYER
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.lstm_out_expanded, W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                print(conv,123)
                # NON-LINEARITY
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #h = tf.nn.relu(conv, name="relu")
                # MAXPOOLING
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # COMBINING POOLED FEATURES
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # #3. DROPOUT LAYER ###################################################################
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("precision"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.precision = self.TP / (self.TP + self.FP)

        with tf.name_scope("recall"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.recall = self.TP / (self.TP + self.FN)

        with tf.name_scope("f1"):
            self.actual=tf.argmax(self.input_y, 1)
            self.TP=tf.count_nonzero(self.predictions * self.actual)
            self.TN=tf.count_nonzero((self.predictions - 1) * (self.actual - 1))
            self.FP = tf.count_nonzero(self.predictions * (self.actual - 1))
            self.FN = tf.count_nonzero((self.predictions - 1) * self.actual)
            self.precision = self.TP / (self.TP + self.FP)
            self.recall = self.TP / (self.TP + self.FN)
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)            

        print("(!!) LOADED LSTM-CNN! :)")
        #embed()


    



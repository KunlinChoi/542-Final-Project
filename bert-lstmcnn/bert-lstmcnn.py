import tensorflow as tf
from model import LSTM_CNN,LSTM
import numpy as np
import os
import time
import datetime
import readvector

def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
        Generates a batch iterator for a dataset.
        """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def train():

    embedding_dim  = 768 
    max_seq_legth = 256
    filter_sizes = [3,4,5] 
    num_filters = 32
    dropout_prob = 0.5
    l2_reg_lambda = 0.0
    batch_size = 64
    num_epochs = 100 
    evaluate_every = 100 
    checkpoint_every = 100 
    num_checkpoints = 0 

    x_train,x_dev,y_train,y_dev,alex,aley = readvector.shuffle()
    #x_train,x_dev,y_train,y_dev,testx,testy,vocab_processor = getvector.getvector()
    print(x_train[0])

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            model = LSTM_CNN(x_train.shape[1],y_train.shape[1],50000,embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
            #model = LSTM(x_train.shape[1],y_train.shape[1],50000,embedding_dim)


            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            dev_summary_op2 = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir2 = os.path.join(out_dir, "summaries", "dev2")
            dev_summary_writer2 = tf.summary.FileWriter(dev_summary_dir2, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            #TRAINING STEP
            def train_step(x_batch, y_batch,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: dropout_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("TRAIN , {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy), file=open("log.txt", "a"))
                if save:
                    train_summary_writer.add_summary(summaries, step)
    
            #EVALUATE MODEL
            def dev_step(x_batch, y_batch, writer=None,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 0.5
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("EVAL {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy), file=open("log.txt", "a"))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)

            def dev_step2(x_batch, y_batch, writer=None,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 0.5
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op2, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("EVAL {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy), file=open("log.txt", "a"))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)

            #CREATE THE BATCHES GENERATOR
            batches = gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
        
            #TRAIN FOR EACH BATCH
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                #print(len(x_batch[0][0]))
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    print(type(y_dev),type(aley),123)
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    dev_step2(alex, aley, writer=dev_summary_writer2)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            #dev_step(testx, testy, writer=dev_summary_writer)



train()


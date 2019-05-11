import tensorflow as tf
from model import LSTM_CNN
import numpy as np
import os
import time
import datetime
import getvector



def gen_batch(data, batch_size, num_epochs, shuffle=True):
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
    embedding_dim  = 32
    max_seq_legth = 128
    filter_sizes = [3,4,5]
    num_filters = 32
    dropout_prob = 0.5
    l2_reg_lambda = 0.0
    batch_size = 64
    num_epochs = 200 
    evaluate_every = 25
    checkpoint_every = 100
    num_checkpoints = 0 

    x_train,x_dev,y_train,y_dev,testx,testy,vocab_processor = getvector.getvector()
    print(x_train[0])
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            model = LSTM_CNN(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim,filter_sizes,num_filters,l2_reg_lambda)



            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            # save model
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            # loss + acc summery
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)
            acc_summary2 = tf.summary.histogram('histogram', model.accuracy)
            #acc_summary = tf.summary.merge([acc_summary1,acc_summary2])
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # test result summery
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # test2 result summery
            #dev_summary_op2 = tf.summary.merge([loss_summary, acc_summary1,acc_summary2])
            #tf.summary.histogram('histogram', acc_summary)
            dev_summary_op2 = tf.summary.merge([loss_summary, acc_summary,acc_summary2])
            dev_summary_dir2 = os.path.join(out_dir, "summaries", "dev2")
            #tf.summary.histogram('histogram', dev_summary_op2)
            dev_summary_writer2 = tf.summary.FileWriter(dev_summary_dir2, sess.graph)
            # Checkpoint saving.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            #global variable
            sess.run(tf.global_variables_initializer())



            #actuall training and calling the model
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

            def dev_step(x_batch, y_batch, writer=None,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 0.5
                }
                step, summaries, loss, accuracy, precision, recall, f1 = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy, model.precision, model.recall, model.f1],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("EVAL {}: step {}, loss {:g}, acc {:g}, prec {:g}, recall {:g}, f1 {:g}".format(time_str, step, loss, accuracy, precision, recall, f1), file=open("log.txt", "a"))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)
                        

            def dev_step2(x_batch, y_batch, writer=None,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 0.5
                }
                step, summaries, loss, accuracy, precision, recall, f1 = sess.run(
                    [global_step, dev_summary_op2, model.loss, model.accuracy, model.precision, model.recall, model.f1],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("EVAL {}: step {}, loss {:g}, acc {:g}, prec {:g}, recall {:g}, f1 {:g}".format(time_str, step, loss, accuracy, precision, recall, f1), file=open("log.txt", "a"))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)

            
            batches = gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
            for batch in batches:
                #print(len(testx[1]),type(testx),"!2")
                #print(len(x_dev[0]),type(x_dev))
                x_batch, y_batch = zip(*batch)
                #print(len(x_batch[0][0]))
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    dev_step2(testx, testy, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            dev_step2(testx, testy, writer=dev_summary_writer)



train()


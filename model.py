import numpy as np
import tensorflow as tf

class LSTM(object):
    
    def __init__(self, hidden_units, num_classes=2, max_sequence_length=50, random_seed=41):
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.random_seed = random_seed
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.X = tf.placeholder(tf.float32, (None, self.max_sequence_length, 1), name='sequences')
            self.X_len = tf.placeholder(tf.int32, (None, 1), name='lengths')
            self.Y = tf.placeholder(tf.float32, (None, self.num_classes), name='parity_labels')

            batch_dim = tf.shape(self.X)[0]
            with tf.variable_scope("LSTM"):
                tf.set_random_seed(self.random_seed)

                cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
                initial_state = cell.zero_state(batch_dim, tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, self.X, initial_state=initial_state, dtype=tf.float32)

            # get last output of rnn
            indices = self.X_len[-1] - 1
            rnn_out = tf.gather(outputs, indices, axis=1)
            self.final_rnn_out = tf.reshape(tf.squeeze(rnn_out), (-1, self.hidden_units))

            with tf.variable_scope("Linear"):
                W = tf.Variable(tf.random_normal((self.hidden_units, self.num_classes), stddev=0.1, seed=self.random_seed))
                b = tf.Variable(tf.zeros((self.num_classes)))
                self.h = tf.matmul(self.final_rnn_out, W) + b

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.h, labels=self.Y))
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

            self.predictions = tf.reshape(tf.squeeze(tf.argmax(self.h, axis=1)), (-1, 1))

def train_model(model, data, labels, lengths,
    epochs=1, batch_size=1, dataset_size=100000, ckpt=False):
    num_batches = dataset_size//batch_size
    ordinal_labels = np.array([np.where(r==1)[0][0] for r in labels])
    
    with model.g.as_default():
        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        if ckpt:
            saver.restore(sess, "/output/model.ckpt")
            print("Model restored.")

        for t in range(epochs):
            print("Epoch {}\n".format(t+1))
            for i in range(0, num_batches, batch_size):
                X_batch = data[i:i+batch_size]
                Y_batch = labels[i:i+batch_size]
                X_len_batch = lengths[i:i+batch_size]
                _  = sess.run(model.train_op, feed_dict={model.X: X_batch, model.Y: Y_batch, model.X_len: X_len_batch})

                if i % 5000 == 0:
                    loss_ = sess.run(model.loss, feed_dict={model.X: data, model.Y: labels, model.X_len: lengths})
                    pred = sess.run(model.predictions, feed_dict={model.X: data, model.Y: labels, model.X_len: lengths})
                    accuracy = np.mean(pred == ordinal_labels)
                    print('iteration: {}, loss: {},  accuracy: {}'.format(i+1, loss_, accuracy))

                    save_path = saver.save(sess, "/output/model.ckpt")
                    print("Model saved in path: %s" % save_path)
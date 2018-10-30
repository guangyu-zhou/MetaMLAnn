import tensorflow as tf
import numpy as np

def load_lap():
    D_path='../data/manifold/genus_dist1.npy'
    D = np.load(D_path).astype(np.float32)
    print("D shape", D.shape)
    # D = np.array([[1,2,3],[0,1,2],[0,1,2]])
    P = np.diag(np.sum(D, axis=1))
    L = D - P
    # print(L, L.shape)
    return L

class Model(object):
    def __init__(self,
            num_features,
            num_labels,
            num_hidden_layers,
            num_hidden_neurons,
            manifold_reg_lambda=1e-7,
            l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_labels], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constant(0.0)

        self.ind_vectors = []
        # individual
        for i in range(num_labels):
            with tf.device('/cpu:0'), tf.variable_scope('ind_{}'.format(i)):
                cur = self.input_x
                for j in range(num_hidden_layers):
                    W = tf.get_variable(
                            'W-{}-{}'.format(i, j),
                            shape=[
                                num_hidden_neurons if j > 0 else num_features,
                                num_hidden_neurons],
                            initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(
                            tf.constant(0.1, shape=[num_hidden_neurons]),
                            name='b-{}-{}'.format(i, j))
                    self.l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
                    cur = tf.nn.dropout(
                            tf.nn.elu(
                                tf.nn.xw_plus_b(cur, W, b),
                                name='pred-{}-{}'.format(i, j)),
                            self.dropout_keep_prob)
                self.ind_vectors.append(cur)


        # shared 
        with tf.device('/cpu:0'), tf.variable_scope('shared'):
            cur = self.input_x
            for j in range(num_hidden_layers):
                W = tf.get_variable(
                        'W-shared-{}'.format(j),
                        shape=[
                            num_hidden_neurons if j > 0 else num_features,
                            num_hidden_neurons],
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(
                        tf.constant(0.1, shape=[num_hidden_neurons]),
                        name='b-shared-{}'.format(j))
                self.l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
                cur = tf.nn.dropout(
                        tf.nn.elu(
                            tf.nn.xw_plus_b(cur, W, b),
                            name='pred-shared-{}'.format(j)),
                        self.dropout_keep_prob)
            self.shared_vector = cur

        self.ind_scores = []
        for i in range(num_labels):
            with tf.device('/cpu:0'), tf.variable_scope('prediction_{}'.format(i)):
                concat_results = tf.concat([self.ind_vectors[i], self.shared_vector], 1)
                W = tf.get_variable(
                        'W-pred-{}'.format(i),
                        shape=[
                            num_hidden_neurons * 2,
                            1],
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(
                        tf.constant(0.1, shape=[1]),
                        name='b-pred-{}'.format(i))
                self.l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
                self.ind_scores.append(
                        tf.sigmoid(
                            tf.nn.xw_plus_b(concat_results, W, b),
                            name='scores-{}'.format(i)))



        self.scores = tf.concat(self.ind_scores, 1)
        print(self.scores.shape)
        # print(self.scores)
        # np.save("score_file.np",self.scores)
        self.predictions = tf.round(self.scores)
        # Lap = load_lap()
        # print("Lap loaded", Lap.shape)
        # Calculate mean cross-entropy loss
        # self.mani = 0.0
        # self.loss_no_mani = 0.0
        with tf.name_scope("loss"):
            losses = -tf.reduce_mean(tf.multiply(self.input_y, tf.log(self.scores + 1e-9))  + tf.multiply(1.0 - self.input_y, tf.log(1.0 - self.scores + 1e-9)))
            # refer to: https://github.com/LayneH/MRCNN/blob/master/MRCNN.py
            # self.mani = -tf.trace(tf.matmul(tf.matmul(self.scores, Lap), self.scores, transpose_b=True))
            # print("mani", mani)
            # self.loss = losses + l2_reg_lambda * self.l2_loss + self.mani*manifold_reg_lambda

            self.loss = losses + l2_reg_lambda * self.l2_loss
            # self.loss_no_mani = losses

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

if __name__ == '__main__':
    Model(25, 555, 1, 64)

import numpy as np
import tensorflow as tf
from DataUtil2 import DataUtil
import copy

class Fism:
    def __init__(self, adv = False, alpha = 0.5):
        self.user_num = 6040
        self.item_num = 3706
        self.K = 64
        self.batch = 1024
        self.max_len = 2313
        self.l2_reg = 5e-6
        self.adv_reg = 5e-6
        self.learning_rate = 8e-4
        self.eps = 1e-3
        self.P = tf.Variable(tf.random_uniform([self.item_num, self.K], minval=-0.1, maxval=0.1))
        self.Q = tf.Variable(tf.random_uniform([self.item_num, self.K], minval=-0.1, maxval=0.1))
        self.delta_P = tf.Variable(tf.zeros(shape = [self.item_num, self.K]), dtype = tf.float32, trainable = False)
        self.delta_Q = tf.Variable(tf.zeros(shape = [self.item_num, self.K]), dtype = tf.float32, trainable = False)
        self.zero_vector = tf.constant(0.0,tf.float32, [1, self.K])
        self.Q = tf.concat([self.Q, self.zero_vector], 0)
        self.bias_u = tf.Variable(tf.random_uniform([self.user_num, 1], minval=-0.1, maxval=0.1))
        self.bias_i = tf.Variable(tf.random_uniform([self.item_num, 1], minval=-0.1, maxval=0.1))
        self.X = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.neighbour = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        #self.alpha = tf.Variable(tf.constant(0.5), dtype=tf.float32)
        self.alpha = alpha
        self.optimizer = None
        self.loss = None
        self.opt_loss = None
        self.logits = None
        self.neighbour_num = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.helper = DataUtil('dataset/ml_train', self.user_num, self.item_num)
        self.adv = adv

    def _logits(self, item_emb, neigh_emb, i):
        sumvec = tf.reduce_sum(neigh_emb, 1)
        # inverse_rated_num = tf.pow(self.rated_num, -tf.constant(self.alpha, tf.float32, [1]))
        inverse_rated_num = tf.pow(self.neighbour_num, -self.alpha)
        inverse_rated_num = tf.reshape(inverse_rated_num, [-1, 1])
        user_repr = tf.multiply(inverse_rated_num, sumvec)
        self.rating = tf.reduce_sum(item_emb * user_repr, axis=1)
        bias_u = tf.reshape(tf.nn.embedding_lookup(self.bias_u, u), [-1])
        bias_i = tf.reshape(tf.nn.embedding_lookup(self.bias_i, i), [-1])
        return tf.nn.sigmoid(self.rating + bias_u + bias_i), bias_u, bias_i

    def inference(self):
        u_i = tf.split(self.X, 2, axis=1)
        i = tf.reshape(u_i[1], [-1])
        item_emb = tf.nn.embedding_lookup(self.P, i)
        neigh_emb = tf.gather(self.Q, self.neighbour)
        logits, bias_u, bias_i = self._logits(self, item_emb, neigh_emb, i)
        return logits, item_emb, neigh_emb, bias_u, bias_i

    def adv_inference(self):
        u_i = tf.split(self.X, 2, axis=1)
        i = tf.reshape(u_i[1], [-1])
        item_plus = tf.nn.embedding_lookup(self.P, i) + tf.nn.embedding_lookup(self.delta_P, i)
        neigh_plus = tf.gather(self.Q, self.neighbour) + tf.gather(self.delta_Q, self.neighbour)
        logits, bias_u, bias_i = self._logits(self, item_emb, neigh_emb, i)
        return logits, item_plus, neigh_plus

    def _loss(self):
        self.out, item_emb, neigh_emb, bias_u, bias_i = self.inference()

        # self.loss = -tf.reduce_sum(
        #     self.Y * tf.log(self.logits + 1e-10) + (1 - self.Y) * tf.log(1 - self.logits + 1e-10)) / self.batch
        self.loss = tf.reduce_sum(tf.square(self.Y - self.out)) / 2 + \
                    self.l2_reg / 2 * (tf.reduce_sum(tf.square(item_emb)) + tf.reduce_sum(tf.square(neigh_emb))) + \
                    self.l2_reg / 2 * (tf.reduce_sum(tf.square(bias_u)) + tf.reduce_sum(tf.square(bias_i)))
        if self.adv:
            self.adv_out, item_plus, neigh_plus = self.adv_inference()
            self.opt_loss = self.loss + self.adv_reg * tf.reduce_sum(tf.square(self.Y - self.adv_out)) / 2 + \
                    self.l2_reg / 2 * (tf.reduce_sum(tf.square(item_plus)) + tf.reduce_sum(tf.square(neigh_plus)))
        else:
            self.opt_loss = self.loss

    def update_delta(self):
        self.grad_P, self.grad_Q = tf.gradients(self.opt_loss, [self.P, self.Q])
        # self.grad_P = tf.stop_gradient(self.grad_P)
        # self.grad_Q = tf.stop_gradient(self.grad_Q)
        self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P, 1) * self.eps)
        self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q, 1) * self.eps)

    def build_graph(self):
        self._inference()
        self._adv_inference()
        self._loss()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
        #                                           initial_accumulator_value=1e-8).minimize(self.loss)

def train(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        max_hit_ratio = 0
        for epoch in range(30):
            print('epoch ' + str(epoch))
            iteration_num = len(self.helper.train) / self.batch
            print('iteration num is:%d' % (iteration_num))
            for i in range(int(iteration_num)):
                if i == 0:
                    x_train, y_train = self.helper.next_batch(self.batch, reset=True)
                else:
                    x_train, y_train = self.helper.next_batch(self.batch)
                y_train = np.reshape(np.array(y_train), (self.batch))
                x_train_neighbours = []
                len_neighbour = []
                # uniform the neighbour length
                for (uid, iid), y_label in zip(x_train, y_train):
                    rated_set = list(self.helper.item_rated_by_user[uid])
                    len_neighbour.append(len(rated_set))
                    if y_label == 1:
                        rated_set.remove(iid)
                    if len(rated_set) < self.max_len:
                        rated_set = list(rated_set)
                        while len(rated_set) < self.max_len:
                            rated_set.append(self.item_num)
                    rated_set = list(rated_set)
                    x_train_neighbours.append(rated_set[:self.max_len])
                len_neighbour = np.array(len_neighbour)
                feed_dict = {
                    self.Y: y_train,
                    self.X: x_train,
                    self.neighbour_num: len_neighbour,
                    self.neighbour: x_train_neighbours}
                if model.adv:
                    sess.run([model.update_P, model.update_Q], feed_dict)
                _, train_loss = sess.run([model.optimizer, model.opt_loss], feed_dict)
                print('loss:%f' % (train_loss))
                # evaluate test data set
                if i % 100 == 0:
                    hit_ratio = evaluate(model, feed_dict, sess)
                    print('hit ratio is %f' % (hit_ratio))
                    if hit_ratio > max_hit_ratio:
                        max_hit_ratio = hit_ratio
                        print('saving best hit ratio:%f' % (hit_ratio))
                        tf.train.Saver().save(sess, './model_save')

def evaluate(model, feed_dict, sess):
    hit_num = 0.0
    for i in range(model.user_num):
        x_test, y_test = model.helper.get_test_batch(i)
        rated_set = model.helper.item_rated_by_user[i]
        neighbour_number = len(rated_set)
        rated_set = list(rated_set)
        if len(rated_set) < model.max_len:
            while len(rated_set) < model.max_len:
                rated_set.append(model.item_num)
        feed_neighbour = []
        for j in range(100):
            feed_neighbour.append(rated_set)
        score = sess.run(model.logits, feed_dict)
        score = np.array(score)
        item_score = []
        for index, t in enumerate(score):
            item_score.append((x_test[index][1], t))
        item_score.sort(key=lambda k: k[1], reverse=True)
        rec_list = set()
        for t in item_score:
            rec_list.add(t[0])
            if len(rec_list) == 10:
                break
        answer = model.helper.test_answer[i]
        # print('answer is %s' % (answer))
        # print(rec_list)
        if int(answer) in rec_list:
            hit_num += 1
    return hit_num / model.user_num

if __name__ == '__main__':
    model = Fism(adv = True)
    model.build_graph()
    train(model)

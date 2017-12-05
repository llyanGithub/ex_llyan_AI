#! /usr/bin/python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Network(object):
    def __init__(self, sizes, learn_rate, epoch, batch):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.batch = batch
        self.bSummary = False
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.sess = tf.InteractiveSession()

        self.weights = []
        self.bias = []

        for i in range(self.num_layers - 1):
            self.weights.append(self.initial_weights([sizes[i], sizes[i+1]]))
            self.bias.append(self.initial_bias([sizes[i+1]]))

        self.x = self.initial_input()
        self.y_ = self.initial_real_output()

        output = self.x
        for i in range(self.num_layers - 1):
            #output = tf.matmul(output, self.weights[i]) + self.bias[i]
            if i == self.num_layers - 2:
                output = self.neurons(output, self.weights[i], self.bias[i], False)
            else:
                output = self.neurons(output, self.weights[i], self.bias[i], True)

        self.y = tf.nn.softmax(output)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cross_entropy)

        self.accuracy = self.initial_accuracy()

        self.sess = tf.InteractiveSession()

        tf.global_variables_initializer().run()

    def neurons(self, in_ ,weights, bias,sigmoid = False):
        if sigmoid == False:
            return tf.matmul(in_, weights) + bias
        return tf.nn.sigmoid(tf.matmul(in_, weights) + bias)

    def add_summary(self, fileName):
        self.summary = tf.summary.scalar(fileName, self.accuracy)
        self.summary_writer = tf.summary.FileWriter(fileName, self.sess.graph)
        self.bSummary = True


    def run(self):
        for i in range(self.epoch):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch)
            if self.bSummary == False:
                self.sess.run(self.train_step, {self.x:batch_xs, self.y_:batch_ys})
            else:
                summary_str, _ = self.sess.run([self.summary, self.train_step], {self.x:batch_xs, self.y_:batch_ys})
                if (i % 10 == 0):
                    self.summary_writer.add_summary(summary_str, i)

    def output(self):
        return self.sess.run(self.accuracy, {self.x: mnist.test.images, self.y_:mnist.test.labels})


    def initial_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy;

    def initial_real_output(self):
        return tf.placeholder(tf.float32, [None, self.sizes[-1]])
    def initial_input(self):
        return tf.placeholder(tf.float32, [None, self.sizes[0]])

    def initial_weights(self, shape):
        return tf.Variable(tf.zeros(shape))

    def initial_bias(self,shape):
        return tf.Variable(tf.zeros(shape))


networkInstance = Network([784, 15, 10], 0.05, 250, 100)
networkInstance.add_summary("network.log")
networkInstance.run()
print(networkInstance.output())

#print(networkInstance.weights)

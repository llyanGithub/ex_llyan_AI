#! /usr/bin/python

import tensorflow as tf
import numpy as np

import re

def get_data():
    f = open("ex_data/lineRegression.txt")
    X = list()
    Y = list()
    for line in f.readlines():
        match = re.match('(?P<X>\d+)  *(?P<Y>[\d.]+)*', line)
        X.append(float(match.group('X')))
        Y.append(float(match.group('Y')))
    f.close()
    m = len(X)
    x = np.array(X)
    y = np.array(Y)

    x.resize(m,1)
    y.resize(m,1)
    return (x, y)

sess = tf.InteractiveSession()

x_data, y_data = get_data()

x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1,1]))

y = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.reduce_sum(tf.pow((y - y_), 2), reduction_indices=[1]))

summary_writer = tf.summary.FileWriter('debug.log', sess.graph)
#tf.summary.scalar("loss", loss)

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#merged_summary_op = tf.summary.merge_all()

tf.global_variables_initializer().run()

for step in range(20):
    train_step.run({x: x_data, y_:y_data}) 

    #summary_str = merged_summary_op.eval()

    #summary_writer.add_summary(summary_str, step)


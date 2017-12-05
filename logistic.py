import tensorflow as tf
import numpy as np

class logistic:
    def __init__(self):
        f = open("ex_data/logistic.txt")
        self.data = list()
        for line in f.readlines():
            self.data.append([ float(x) for x in line.strip().split(',')])

        class logistic_data:
            def __init__(self, data):
                tmp_data = list()
                tmp_labels = list()
                for item in data:
                    tmp_data.append(item[0:-2]) 
                    tmp_labels.append(item[-1])

                self.cur = 0
                self.data = np.array(tmp_data)
                self.labels = np.array([[x] for x in tmp_labels])
                #self.labels.resize((self.labels.shape, 1))


                return
            
            def data(self):
                return self.data

            def labels(self):
                return self.labels

            def next_batch(self, num):
                if self.cur + num > len(self.data):
                    batch_xs = self.data[self.cur:-1]
                    batch_ys = self.labels[self.cur:-1]
                    self.cur = 0
                else:
                    batch_xs = self.data[self.cur:self.cur + num]
                    batch_ys = self.labels[self.cur: self.cur + num]
                    self.cur += num

                return (batch_xs, batch_ys)

            
        self.dataItem = len(self.data)
        self.trainItem = int(self.dataItem * 0.8)
        self.testItem = self.dataItem - self.trainItem

        self.train = logistic_data(self.data[0:self.trainItem])
        self.test = logistic_data(self.data[self.trainItem:self.dataItem])

        return

logFile = "logistic.log"

data = logistic()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 7])

w = tf.Variable(tf.zeros([7, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.nn.sigmoid(tf.matmul(x, w) + b)
y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#summary_writer = tf.summary.FileWriter(logFile, sess.graph)

tf.global_variables_initializer().run()

for i in range(200):
    batch_xs, batch_ys = data.train.next_batch(20)
    train_step.run({x: batch_xs, y_:batch_ys})
    #print(sess.run(w))
    #print(sess.run(b))
    print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}))
    #print(sess.run(w))
    #print(sess.run(b))
    #print(sess.run(cross_entropy, feed_dict={x: data.test.data, y_:data.test.labels}))

correct_prediction = tf.subtract(y, y_)

accuracy = tf.reduce_mean(correct_prediction)

print(accuracy.eval({x: data.test.data, y_:data.test.labels}))


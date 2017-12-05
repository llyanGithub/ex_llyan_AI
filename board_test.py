import tensorflow as tf

a = tf.Variable(3.3)
b = tf.Variable(2.3)
c = tf.add(a,b)
d = tf.add(a,1)
update = tf.assign(a,d)

sess = tf.InteractiveSession()

scalar = tf.summary.scalar("update",update)
summary_writer = tf.summary.FileWriter("update.log", sess.graph)

tf.global_variables_initializer().run()

for i in range(10):
    update.eval()
    summary_str = scalar.eval()
    summary_writer.add_summary(summary_str)
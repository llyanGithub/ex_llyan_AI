import tensorflow as tf

cross_entropy is tensor
tf.summary.scalar("cross_entropy", cross_entropy)

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/cnn1.log', sess.graph)

summary_str = merged_summary_op.eval()

summary_writer.add_summary(summary_str, step)

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os, wget
import tensorflow as tf
import numpy as np

LOG_DIR = "./logs"
if (os.path.isdir(LOG_DIR) == False):
    os.mkdir(LOG_DIR, mode=755)


mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

WIDTH = 28
HEIGHT = 28
TRAIN_SET_SIZE = (int)(mnist.train.images.size / (WIDTH * HEIGHT))
TEST_SET_SIZE = (int)(mnist.test.images.size / (WIDTH * HEIGHT))

print(str(TRAIN_SET_SIZE) + " train sets contained")
print(str(TEST_SET_SIZE) + " test sets contained")


PATH_TO_SPRITE_IMAGE = "./mnist_10k_sprite.png"
if (os.path.isfile(PATH_TO_SPRITE_IMAGE) == False):
    mnist_sprite_image_url = "https://www.tensorflow.org/images/mnist_10k_sprite.png"
    wget.download(mnist_sprite_image_url, out=PATH_TO_SPRITE_IMAGE)


with tf.name_scope('model') as scope :
    X = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT], name="X")
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")

    conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)

    flat = tf.contrib.layers.flatten(pool3)

    dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense4, units=10)

with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

with tf.name_scope('train') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.name_scope('eval') as h1scope:
    correct_predition = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)


test_images = mnist.test.images[:TEST_SET_SIZE]
embedding_var = tf.Variable(test_images, name='embedding')


config = projector.ProjectorConfig()


embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name


np.savetxt(os.path.join(LOG_DIR, 'metadata.tsv'), mnist.test.labels[:TEST_SET_SIZE], fmt='%d')


embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE


embedding.sprite.single_image_dim.extend([WIDTH, HEIGHT])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
projector.visualize_embeddings(summary_writer, config)
saver = tf.train.Saver()

for i in range(1000) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict = {X : batch_xs, Y : batch_ys})
    summary_writer.add_summary(summary, i)
    if i % 100 == 0:
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)


print(sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))
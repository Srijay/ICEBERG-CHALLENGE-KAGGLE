import tensorflow as tf
import json
from pprint import pprint
import numpy as np
import random

def cnn(x):

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 75, 75, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([23104, 512])
    b_fc1 = bias_variable([512])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 23104])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  #with tf.name_scope('dropout'):
    #h_fc1_drop = tf.nn.dropout(h_fc1, tf.constant(0.5)) #0.5 while training & 1 while testing

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([512, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2 #change h_fc1 to h_fc1_drop after applying dropout
  return y_conv

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

print("Welcome to iceberg challenge")

traindata = json.load(open('data/processed/train.json'))

trainband1 = []
trainband2 = []
trainDatalabels = []

for data in traindata:
    trainband1.append(data["band_1"])
    trainband2.append(data["band_2"])
    trainDatalabels.append(int(data["is_iceberg"]))

trainband1 = np.array(trainband1)
trainband2 = np.array(trainband2)
trainDatalabels = np.array(trainDatalabels)

onehot = np.zeros((len(trainDatalabels),2))
onehot[np.arange(len(trainDatalabels)),trainDatalabels] = 1
trainlabels = np.array(onehot,dtype=np.float32)

band1tf =  tf.placeholder(tf.float32, [None, 5625])
band2tf =  tf.placeholder(tf.float32, [None, 5625])
labeltf = tf.placeholder(tf.float32, [None, 2])

band1cnn = cnn(band1tf)
band1coef = weight_variable([2])
band2cnn = cnn(band2tf)
band2coef = weight_variable([2])
icebergProb = band1coef*band1cnn + band2coef*band2cnn

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labeltf,logits=icebergProb)

cross_entropy = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(icebergProb, 1), tf.argmax(labeltf, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

output = tf.transpose(icebergProb)[1]


sess = tf.Session()

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
model_path = "./model.ckpt";

Iterations = 2000

for i in range(Iterations):

    minibatchIds = random.sample(range(0, len(trainlabels)), 20)

    miniBatchband1 = [trainband1[k] for k in minibatchIds]
    miniBatchband2 = [trainband2[k] for k in minibatchIds]
    miniBatchLabels = [trainlabels[k] for k in minibatchIds]

    if i % 20 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={
            band1tf: miniBatchband1, band2tf: miniBatchband2,labeltf: miniBatchLabels})
        saver.save(sess, model_path)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_step,feed_dict={ band1tf: miniBatchband1, band2tf: miniBatchband2,labeltf: miniBatchLabels})

saver.restore(sess,model_path)
print("Model restored")


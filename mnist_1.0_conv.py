import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

# The model is:
# Convolutional Neural Network
# -> [100 28x28 images input]
# -> [5x5 patch, 1 input, 4 output], stride 1
# -> [4x4 patch, 4 input, 8 output], stride 2
# -> [4x4 patch, 8 input, 12 output], stride 2
# -> [200 fully connected neurons]
# -> [10 output neurons]

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# variable learning rate
lr = tf.placeholder(tf.float32)

# Correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# Input layer
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # input of a batch of 100 images each 28x28

# 1st convolutional layer - 28x28x4
P = 5 # filter patch size in px
I = 1 # input channels = output channels of previous layer
O = 4 # new output channels
S = 1 # strides
W1 = tf.Variable(tf.truncated_normal([P, P, I, O], stddev=0.1))
B1 = tf.Variable(tf.truncated_normal([O], stddev=0.1)) # biases for output channels
Y1_cnv = tf.nn.conv2d(X, W1, strides=[1, S, S, 1], padding='SAME')
Y1 = tf.nn.relu(Y1_cnv + B1)

# 2nd convolutional layer - 14x14x8
P = 4 # filter patch size in px
I = O # input channels = output channels of previous layer
O = 8 # new output channels
S = 2 # strides
W2 = tf.Variable(tf.truncated_normal([P, P, I, O], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([O], stddev=0.1))
Y2_cnv = tf.nn.conv2d(Y1, W2, strides=[1, S, S, 1], padding="SAME")
Y2 = tf.nn.relu(Y2_cnv + B2)

# 3nd convolutional layer - 7x7x12
P = 4 # filter patch size in px
I = O # input channels = output channels of previous layer
O = 12 # new output channels
S = 2 # strides
W3 = tf.Variable(tf.truncated_normal([P, P, I, O], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([O], stddev=0.1))
Y3_cnv = tf.nn.conv2d(Y2, W3, strides=[1, S, S, 1], padding="SAME")
Y3 = tf.nn.relu(Y3_cnv + B3)

# Flatten output
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * O])

# Fully connected layer - [7*7*12x200]
N = 200 # neurons
W4 = tf.Variable(tf.truncated_normal([7 * 7 * O, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N]))
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

# Output layer
W = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B = tf.Variable(tf.zeros([10]))
Y = tf.nn.softmax(tf.matmul(Y4, W) + B)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, variable learning rate
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(B, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    lrmin = 0.0001
    lrmax = 0.003
    decay = 2000
    learn_rate = lrmin + (lrmax-lrmin) * math.exp(-i/decay)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learn_rate})


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

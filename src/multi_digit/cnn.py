import time
import numpy as np
import tensorflow as tf

from src.multi_digit.svhn import SVHNMulti

# Parameters
learning_rate = 0.001
iterations = 15000
batch_size = 50
display_step = 1000

# Network Parameters
channels = 3
image_size = 64
n_classes = 11
n_labels = 6
dropout = 0.75
depth_1 = 32
depth_2 = 32
depth_3 = 64
depth_4 = 128
hidden = 256
filter_size = 5
normalization_offset = 0.0  # beta
normalization_scale = 1.0  # gamma
normalization_epsilon = 0.001  # epsilon


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def convolution(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Load data
svhn = SVHNMulti("../../res/processed/normalized")
train_data, train_labels = svhn.load_data("train")
train_examples = len(train_data)


# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.float32, [None, n_labels, n_classes])

# Weights & Biases
weights = {
    "layer1": weight_variable([filter_size, filter_size, channels, depth_1]),
    "layer2": weight_variable([filter_size, filter_size, depth_1, depth_2]),
    "layer3": weight_variable([filter_size, filter_size, depth_2, depth_3]),
    "layer4": weight_variable([filter_size, filter_size, depth_3, depth_4]),
    "layer5": weight_variable([image_size // 16 * image_size // 16 * depth_4, hidden]),
    "digit1": weight_variable([hidden, n_classes]),
    "digit2": weight_variable([hidden, n_classes]),
    "digit3": weight_variable([hidden, n_classes]),
    "digit4": weight_variable([hidden, n_classes]),
    "digit5": weight_variable([hidden, n_classes])
}

biases = {
    "layer1": bias_variable([depth_1]),
    "layer2": bias_variable([depth_2]),
    "layer3": bias_variable([depth_3]),
    "layer4": bias_variable([depth_4]),
    "layer5": bias_variable([hidden]),
    "digit1": bias_variable([n_classes]),
    "digit2": bias_variable([n_classes]),
    "digit3": bias_variable([n_classes]),
    "digit4": bias_variable([n_classes]),
    "digit5": bias_variable([n_classes])
}


def normalize(x):
    """ Applies batch normalization """
    mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, normalization_offset, normalization_scale,
                                     normalization_epsilon)


def cnn(x):
    # Batch normalization
    # x = normalize(x)

    # Convolution 1 -> RELU -> Max Pool
    convolution1 = convolution(x, weights["layer1"])
    relu1 = tf.nn.relu(convolution1 + biases["layer1"])
    maxpool1 = max_pool(relu1)

    # Convolution 2 -> RELU -> Max Pool
    convolution2 = convolution(maxpool1, weights["layer2"])
    relu2 = tf.nn.relu(convolution2 + biases["layer2"])
    maxpool2 = max_pool(relu2)

    # Convolution 3 -> RELU -> Max Pool
    convolution3 = convolution(maxpool2, weights["layer3"])
    relu3 = tf.nn.relu(convolution3 + biases["layer3"])
    maxpool3 = max_pool(relu3)

    # Convolution 4 -> RELU -> Max Pool
    convolution4 = convolution(maxpool3, weights["layer4"])
    relu4 = tf.nn.relu(convolution4 + biases["layer4"])
    maxpool4 = max_pool(relu4)

    # Fully Connected Layer
    shape = maxpool4.get_shape().as_list()
    reshape = tf.reshape(maxpool4, [-1, shape[1] * shape[2] * shape[3]])
    fc = tf.nn.relu(tf.matmul(reshape, weights["layer5"]) + biases["layer5"])

    # Dropout Layer
    keep_prob_constant = tf.placeholder(tf.float32)
    dropout_layer = tf.nn.dropout(fc, keep_prob_constant)

    logit1 = tf.matmul(dropout_layer, weights["digit1"]) + biases["digit1"]
    logit2 = tf.matmul(dropout_layer, weights["digit2"]) + biases["digit2"]
    logit3 = tf.matmul(dropout_layer, weights["digit3"]) + biases["digit3"]
    logit4 = tf.matmul(dropout_layer, weights["digit4"]) + biases["digit4"]
    logit5 = tf.matmul(dropout_layer, weights["digit5"]) + biases["digit5"]

    return logit1, logit2, logit3, logit4, logit5, keep_prob_constant


# Build the graph for the deep net
digit1, digit2, digit3, digit4, digit5, keep_prob = cnn(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 1, :], logits=digit1)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 2, :], logits=digit2)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 3, :], logits=digit3)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 4, :], logits=digit4)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 5, :], logits=digit5))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
prediction = tf.stack([tf.nn.softmax(digit1),
                       tf.nn.softmax(digit2),
                       tf.nn.softmax(digit3),
                       tf.nn.softmax(digit4),
                       tf.nn.softmax(digit5)])
# prediction = tf.stack([digit1, digit2, digit3, digit4, digit5])
prediction = tf.transpose(prediction, [1, 0, 2])
actual = tf.transpose(tf.stack([Y[:, 1, :], Y[:, 2, :], Y[:, 3, :], Y[:, 4, :], Y[:, 5, :]]), [1, 0, 2])
correct_prediction = tf.equal(tf.argmax(prediction, 2), tf.argmax(actual, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
accuracy_part = tf.reduce_mean(accuracy)
accuracy_whole = tf.reduce_mean(tf.cast(tf.equal(accuracy, tf.constant(1.0)), tf.float32))

# Start counting execution time
start_time = time.time()

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    # Variables useful for batch creation
    start = 0
    end = 0

    # The accuracy and loss for every iteration in train and test set
    train_accuracies = []
    train_losses = []
    test_accuracies_part = []
    test_accuracies_whole = []
    test_losses = []

    for i in range(iterations):
        # Construct the batch
        if start == train_examples:
            start = 0
        end = min(train_examples, start + batch_size)

        batch_x = train_data[start:end]
        batch_y = train_labels[start:end]

        start = end

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if (i + 1) % display_step == 0 or i == 0:
            _accuracy_part, _accuracy_whole, _cost = sess.run([accuracy_part, accuracy_whole, cost],
                                                              feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step: {0:6d}, Training Accuracy Whole: {1:5f}, Training Accuracy Part: {2:5f}, Batch Loss: {3:5f}"
                  .format(i + 1, _accuracy_whole, _accuracy_part, _cost))
            train_accuracies.append(_accuracy_whole)
            train_losses.append(_cost)

    # Free space by deleting training data
    train_data = []
    train_labels = []

    # Test the model by measuring it's accuracy
    test_data, test_labels = svhn.load_data("test")
    test_examples = len(test_data)

    test_iterations = test_examples // batch_size + 1
    for i in range(test_iterations):
        batch_x, batch_y = (test_data[i * batch_size:(i + 1) * batch_size],
                            test_labels[i * batch_size:(i + 1) * batch_size])
        _accuracy_part, _accuracy_whole, _cost = sess.run([accuracy_part, accuracy_whole, cost],
                                                          feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        test_accuracies_part.append(_accuracy_part)
        test_accuracies_whole.append(_accuracy_whole)
        test_losses.append(_cost)
    print("Mean Test Accuracy Part: {0:5f}, Mean Test Accuracy Whole: {1:5f}, Mean Test Loss: {2:5f}"
          .format(np.mean(test_accuracies_part), np.mean(test_accuracies_whole), np.mean(test_losses)))

    # print execution time
    print("Execution time in seconds: " + str(time.time() - start_time))

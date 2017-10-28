import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.multi_digit.svhn import SVHNMulti

# Parameters
learning_rate = 0.001
iterations = 10000
batch_size = 50
display_step = 1000

# Use of extra dataset
use_extra = True

# Use only one of the below flags as true
plain = True
normalized = False
gray = False

# Data Directory where the processed data reside
data_dir = "D:/res/processed/"

# Network Parameters
channels = 3
image_size = 64
n_classes = 11
n_labels = 3
dropout = 0.85
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


# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.int32, [None, n_labels, n_classes])

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
    x = normalize(x)

    # Convolution 1 -> RELU -> Max Pool
    convolution1 = convolution(x, weights["layer1"])
    relu1 = tf.nn.relu(convolution1 + biases["layer1"])
    maxpool1 = avg_pool(relu1)

    # Convolution 2 -> RELU -> Max Pool
    convolution2 = convolution(maxpool1, weights["layer2"])
    relu2 = tf.nn.relu(convolution2 + biases["layer2"])
    maxpool2 = avg_pool(relu2)

    # Convolution 3 -> RELU -> Max Pool
    convolution3 = convolution(maxpool2, weights["layer3"])
    relu3 = tf.nn.relu(convolution3 + biases["layer3"])
    maxpool3 = avg_pool(relu3)

    # Convolution 4 -> RELU -> Max Pool
    convolution4 = convolution(maxpool3, weights["layer4"])
    relu4 = tf.nn.relu(convolution4 + biases["layer4"])
    maxpool4 = avg_pool(relu4)

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

    return logit1, logit2, logit3, keep_prob_constant


def visualize_results(train_accuracies, train_losses, test_accuracies, test_losses, test_iterations, train_examples):
    # Plot batch accuracy and loss for both train and test sets
    plt.style.use("ggplot")
    fig, ax = plt.subplots(2, 2)

    # Train Accuracy
    ax[0, 0].set_title("Train Accuracy per Batch")
    ax[0, 0].set_xlabel("Batch")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].set_ylim([0, 1.05])
    ax[0, 0].plot(range(0, iterations + 1, display_step), train_accuracies, linewidth=1, color="darkgreen")

    # Train Loss
    ax[0, 1].set_title("Train Loss per Batch")
    ax[0, 1].set_xlabel("Batch")
    ax[0, 1].set_ylabel("Loss")
    ax[0, 1].set_ylim([0, max(train_losses)])
    ax[0, 1].plot(range(0, iterations + 1, display_step), train_losses, linewidth=1, color="darkred")

    # Test Accuracy
    ax[1, 0].set_title("Test Accuracy per Batch")
    ax[1, 0].set_xlabel("Batch")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].set_ylim([0, 1.05])
    ax[1, 0].plot(range(0, test_iterations), test_accuracies, linewidth=1, color="darkgreen")

    # Test Loss
    ax[1, 1].set_title("Test Loss per Batch")
    ax[1, 1].set_xlabel("Batch")
    ax[1, 1].set_ylabel("Loss")
    ax[1, 1].set_ylim([0, max(test_losses)])
    ax[1, 1].plot(range(0, test_iterations), test_losses, linewidth=1, color="darkred")

    for i in range(1, iterations, train_examples // batch_size):
        ax[0, 0].axvline(x=i, ymin=0, ymax=1.05, linewidth=2, color="orange", label="skdlhv")
        ax[0, 1].axvline(x=i, ymin=0, ymax=max(train_losses), linewidth=2, color="orange")

    plt.show()


def load_data():
    if normalized:
        directory = "normalized"
    elif gray:
        directory = "gray"
    else:
        directory = "plain"

    directory = directory + "_" + str(n_labels)

    svhn = SVHNMulti(data_dir + directory)
    train_data, train_labels = svhn.load_data("train")
    test_data, test_labels = svhn.load_data("test")

    if use_extra:
        extra_data, extra_labels = svhn.load_data("extra")
        train_data = np.append(train_data, extra_data, axis=0)
        train_labels = np.append(train_labels, extra_labels, axis=0)

    return train_data, train_labels, test_data, test_labels


def main():
    # Load data
    train_data, train_labels, test_data, test_labels = load_data()
    train_examples = len(train_data)
    test_examples = len(test_data)

    # Build the graph for the deep net
    digit1, digit2, digit3, keep_prob = cnn(X)

    # Cost is composed by adding all losses of each and every digit
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 0, :], logits=digit1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 1, :], logits=digit2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 2, :], logits=digit3))
    cost = loss1 + loss2 + loss3

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Stack predictions as one large array
    prediction = tf.stack([digit1, digit2, digit3])

    # Transpose prediction array as needed
    prediction = tf.transpose(prediction, [1, 0, 2])

    # Stack and transpose actual labels, in the same way as predictions
    actual = tf.transpose(tf.stack([Y[:, 0, :], Y[:, 1, :], Y[:, 2, :]]), [1, 0, 2])

    # Compute equality vectors
    correct_prediction = tf.equal(tf.argmax(prediction, 2), tf.argmax(actual, 2))

    # Calculate mean accuracy among 1st dimension
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)

    # Accuracy of predicting any digit in the images
    accuracy_single = tf.reduce_mean(accuracy)

    # Accuracy of the predicting all numbers in an image
    accuracy_multi = tf.reduce_mean(tf.cast(tf.equal(accuracy, tf.constant(1.0)), tf.float32))

    # Start counting execution time
    start_time = time.time()

    with tf.Session() as sess:
        # Initialize Tensorflow variables
        sess.run(tf.global_variables_initializer())

        # Variables useful for batch creation
        start = 0

        # The accuracy and loss for every iteration in train and test set
        train_accuracies = []
        train_losses = []
        test_accuracies_single = []
        test_accuracies_multi = []
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
                _accuracy_single, _accuracy_multi, _cost = sess.run([accuracy_single, accuracy_multi, cost],
                                                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
                print("Step: {0:6d}, "
                      "Training Accuracy Whole: {1:5f}, "
                      "Training Accuracy Part: {2:5f}, "
                      "Batch Loss: {3:5f}".format(i + 1, _accuracy_multi, _accuracy_single, _cost))

                train_accuracies.append(_accuracy_multi)
                train_losses.append(_cost)

        # Test the model by measuring it's accuracy
        test_iterations = test_examples // batch_size + 1
        for i in range(test_iterations):
            batch_x, batch_y = (test_data[i * batch_size:(i + 1) * batch_size],
                                test_labels[i * batch_size:(i + 1) * batch_size])
            _accuracy_single, _accuracy_multi, _cost = sess.run([accuracy_single, accuracy_multi, cost],
                                                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            test_accuracies_single.append(_accuracy_single)
            test_accuracies_multi.append(_accuracy_multi)
            test_losses.append(_cost)
        print("Mean Test Accuracy Part: {0:5f}, "
              "Mean Test Accuracy Whole: {1:5f}, "
              "Mean Test Loss: {2:5f}".format(np.mean(test_accuracies_single),
                                              np.mean(test_accuracies_multi),
                                              np.mean(test_losses)))

        # print execution time
        print("Execution time in seconds: " + str(time.time() - start_time))

        # Visualize results
        visualize_results(train_accuracies, train_losses, test_accuracies_multi, test_losses, test_iterations,
                          train_examples)


if __name__ == '__main__':
    main()

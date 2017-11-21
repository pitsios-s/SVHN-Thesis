import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.multi_digit.svhn import SVHNMulti

# Parameters
learning_rate = 0.001
iterations = 20000
batch_size = 50
display_step = 1000

# Use of extra dataset
use_extra = True

# Use only one of the below flags as true
plain = False
normalized = False
gray = True

# Data Directory where the processed data reside
data_dir = "D:/res/processed/"

# Network Parameters
channels = 1
image_size = 64
n_classes = 11
n_labels = 5
dropout = 0.8
depth_1 = 16
depth_2 = 32
depth_3 = 64
depth_4 = 128
depth_5 = 128
hidden = 128
filter_size = 5
normalization_offset = 0.0  # beta
normalization_scale = 1.0  # gamma
normalization_epsilon = 0.001  # epsilon

# Tensorboard parameters
train_log_dir = "../../logs/multi/train"
test_log_dir = "../../logs/multi/test"


def weight_variable(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def bias_variable(shape, name=None):
    return tf.Variable(tf.constant(1.0, shape=shape), name=name)


def convolution(x, w, name=None):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME", name=name)


def max_pool(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)


def avg_pool(x, name=None):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)


# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.int32, [None, n_labels, n_classes])

# Weights & Biases
weights = {
    "layer1": weight_variable([filter_size, filter_size, channels, depth_1], name="weight_layer_1"),
    "layer2": weight_variable([filter_size, filter_size, depth_1, depth_2], name="weight_layer_2"),
    "layer3": weight_variable([filter_size, filter_size, depth_2, depth_3], name="weight_layer_3"),
    "layer4": weight_variable([filter_size, filter_size, depth_3, depth_4], name="weight_layer_4"),
    "layer5": weight_variable([filter_size, filter_size, depth_4, depth_5], name="weight_layer_5"),
    "layer6": weight_variable([image_size // 32 * image_size // 32 * depth_5, hidden], name="weight_dropout"),
    "digit1": weight_variable([hidden, n_classes], name="weight_digit_1"),
    "digit2": weight_variable([hidden, n_classes], name="weight_digit_2"),
    "digit3": weight_variable([hidden, n_classes], name="weight_digit_3"),
    "digit4": weight_variable([hidden, n_classes], name="weight_digit_4"),
    "digit5": weight_variable([hidden, n_classes], name="weight_digit_5")
}

biases = {
    "layer1": bias_variable([depth_1], name="bias_layer_1"),
    "layer2": bias_variable([depth_2], name="bias_layer_2"),
    "layer3": bias_variable([depth_3], name="bias_layer_3"),
    "layer4": bias_variable([depth_4], name="bias_layer_4"),
    "layer5": bias_variable([depth_5], name="bias_layer_5"),
    "layer6": bias_variable([hidden], name="bias_dropout"),
    "digit1": bias_variable([n_classes], name="bias_digit_1"),
    "digit2": bias_variable([n_classes], name="bias_digit_2"),
    "digit3": bias_variable([n_classes], name="bias_digit_3"),
    "digit4": bias_variable([n_classes], name="bias_digit_4"),
    "digit5": bias_variable([n_classes], name="bias_digit_5")
}


def normalize(x):
    """ Applies batch normalization """
    mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, normalization_offset, normalization_scale,
                                     normalization_epsilon)


def cnn(x):
    # Batch normalization
    x = normalize(x)

    # Convolution 1 -> RELU -> Pool
    with tf.name_scope("convolution_layer_1"):
        convolution1 = convolution(x, weights["layer1"], name="conv_1")
        relu1 = tf.nn.relu(convolution1 + biases["layer1"], name="relu_1")
        pool1 = avg_pool(relu1, name="avg_pool_1")

    # Convolution 2 -> RELU -> Pool
    with tf.name_scope("convolution_layer_2"):
        convolution2 = convolution(pool1, weights["layer2"], name="conv_2")
        relu2 = tf.nn.relu(convolution2 + biases["layer2"], name="relu_2")
        pool2 = avg_pool(relu2, name="avg_pool_2")

    # Convolution 3 -> RELU -> Pool
    with tf.name_scope("convolution_layer_3"):
        convolution3 = convolution(pool2, weights["layer3"], name="conv_3")
        relu3 = tf.nn.relu(convolution3 + biases["layer3"], name="relu_3")
        pool3 = avg_pool(relu3, name="avg_pool_3")

    # Convolution 4 -> RELU -> Pool
    with tf.name_scope("convolution_layer_4"):
        convolution4 = convolution(pool3, weights["layer4"], name="conv_4")
        relu4 = tf.nn.relu(convolution4 + biases["layer4"], name="relu_4")
        pool4 = avg_pool(relu4, name="avg_pool_4")

    # Convolution 5 -> RELU -> Pool
    with tf.name_scope("convolution_layer_5"):
        convolution5 = convolution(pool4, weights["layer5"], name="conv_5")
        relu5 = tf.nn.relu(convolution5 + biases["layer5"], name="relu_5")
        pool5 = avg_pool(relu5, name="avg_pool_5")

    # Fully Connected Layer
    with tf.name_scope("fully_connected_layer"):
        shape = pool5.get_shape().as_list()
        reshape = tf.reshape(pool5, [-1, shape[1] * shape[2] * shape[3]])
        fc = tf.nn.relu(tf.matmul(reshape, weights["layer6"]) + biases["layer6"])

    # Dropout Layer
    with tf.name_scope("dropout"):
        keep_prob_constant = tf.placeholder(tf.float32)
        dropout_layer = tf.nn.dropout(fc, keep_prob_constant)

    # Output Layer
    with tf.name_scope("output"):
        logit1 = tf.matmul(dropout_layer, weights["digit1"]) + biases["digit1"]
        logit2 = tf.matmul(dropout_layer, weights["digit2"]) + biases["digit2"]
        logit3 = tf.matmul(dropout_layer, weights["digit3"]) + biases["digit3"]
        logit4 = tf.matmul(dropout_layer, weights["digit4"]) + biases["digit4"]
        logit5 = tf.matmul(dropout_layer, weights["digit5"]) + biases["digit5"]

    return logit1, logit2, logit3, logit4, logit5, keep_prob_constant


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


def clear_old_logs():
    """ Clears any previous log files """
    if tf.gfile.Exists(train_log_dir):
        tf.gfile.DeleteRecursively(train_log_dir)
    tf.gfile.MakeDirs(train_log_dir)

    if tf.gfile.Exists(test_log_dir):
        tf.gfile.DeleteRecursively(test_log_dir)
    tf.gfile.MakeDirs(test_log_dir)


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

    # Clear existing logs
    clear_old_logs()

    # Build the graph for the deep net
    digit1, digit2, digit3, digit4, digit5, keep_prob = cnn(X)

    # Cost is composed by adding all losses of each and every digit
    with tf.name_scope("loss"):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 0, :], logits=digit1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 1, :], logits=digit2))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 2, :], logits=digit3))
        loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 3, :], logits=digit4))
        loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y[:, 4, :], logits=digit5))
        cost = loss1 + loss2 + loss3 + loss4 + loss5
    tf.summary.scalar("loss", cost)

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope("accuracy"):
        with tf.name_scope("prediction"):
            # Stack predictions as one large array
            prediction = tf.stack([digit1, digit2, digit3, digit4, digit5])

            # Transpose prediction array as needed
            prediction = tf.transpose(prediction, [1, 0, 2])
        with tf.name_scope("actual_values"):
            # Stack and transpose actual labels, in the same way as predictions
            actual = tf.transpose(tf.stack([Y[:, 0, :], Y[:, 1, :], Y[:, 2, :], Y[:, 3, :], Y[:, 4, :]]), [1, 0, 2])

        with tf.name_scope("correct_prediction"):
            # Compute equality vectors
            correct_prediction = tf.equal(tf.argmax(prediction, 2), tf.argmax(actual, 2))

            # Calculate mean accuracy among 1st dimension
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)

        with tf.name_scope("accuracy_single_digit"):
            # Accuracy of predicting any digit in the images
            accuracy_single = tf.reduce_mean(accuracy)

        with tf.name_scope("accuracy_multi_digit"):
            # Accuracy of the predicting all numbers in an image
            accuracy_multi = tf.reduce_mean(tf.cast(tf.equal(accuracy, tf.constant(1.0)), tf.float32))
    tf.summary.scalar("accuracy_single_digit", accuracy_single)
    tf.summary.scalar("accuracy_multi_digit", accuracy_multi)

    # Start counting execution time
    start_time = time.time()

    with tf.Session() as sess:
        # Initialize Tensorflow variables
        sess.run(tf.global_variables_initializer())

        # Writers for storing tensorboard statistics
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

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
                _accuracy_single, _accuracy_multi, _cost, _summary = \
                    sess.run([accuracy_single, accuracy_multi, cost, merged],
                             feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
                print("Step: {0:6d}, "
                      "Training Accuracy Multi: {1:5f}, "
                      "Training Accuracy Single: {2:5f}, "
                      "Batch Loss: {3:5f}".format(i + 1, _accuracy_multi, _accuracy_single, _cost))

                train_accuracies.append(_accuracy_multi)
                train_losses.append(_cost)
                train_writer.add_summary(_summary, i)

        # Test the model by measuring it's accuracy
        test_iterations = test_examples // batch_size + 1
        for i in range(test_iterations):
            batch_x, batch_y = (test_data[i * batch_size:(i + 1) * batch_size],
                                test_labels[i * batch_size:(i + 1) * batch_size])
            _accuracy_single, _accuracy_multi, _cost, _summary = \
                sess.run([accuracy_single, accuracy_multi, cost, merged],
                         feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            test_accuracies_single.append(_accuracy_single)
            test_accuracies_multi.append(_accuracy_multi)
            test_losses.append(_cost)
            test_writer.add_summary(_summary, i)
        print("Mean Test Accuracy Single: {0:5f}, "
              "Mean Test Accuracy Multi: {1:5f}, "
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

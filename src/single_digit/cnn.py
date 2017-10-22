import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.single_digit.svhn import SVHN

# Parameters
learning_rate = 0.001
iterations = 50000
batch_size = 50
display_step = 1000

# Network Parameters
channels = 3
image_size = 32
n_classes = 10
dropout = 0.85
hidden = 128
depth_1 = 32
depth_2 = 64
depth_3 = 128
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
svhn = SVHN("../../res/cropped", n_classes, use_extra=True, gray=False, normalize=False)

# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.float32, [None, n_classes])


# Weights & Biases
weights = {
    "layer1": weight_variable([filter_size, filter_size, channels, depth_1]),
    "layer2": weight_variable([filter_size, filter_size, depth_1, depth_2]),
    "layer3": weight_variable([filter_size, filter_size, depth_2, depth_3]),
    "layer4": weight_variable([image_size // 8 * image_size // 8 * depth_3, hidden]),
    "layer5": weight_variable([hidden, n_classes])
}

biases = {
    "layer1": bias_variable([depth_1]),
    "layer2": bias_variable([depth_2]),
    "layer3": bias_variable([depth_3]),
    "layer4": bias_variable([hidden]),
    "layer5": bias_variable([n_classes])
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

    # Fully Connected Layer
    shape = maxpool3.get_shape().as_list()
    reshape = tf.reshape(maxpool3, [-1, shape[1] * shape[2] * shape[3]])
    fc = tf.nn.relu(tf.matmul(reshape, weights["layer4"]) + biases["layer4"])

    # Dropout Layer
    keep_prob_constant = tf.placeholder(tf.float32)
    dropout_layer = tf.nn.dropout(fc, keep_prob_constant)

    return tf.matmul(dropout_layer, weights["layer5"]) + biases["layer5"], keep_prob_constant


# Build the graph for the deep net
y_conv, keep_prob = cnn(X)

# The cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))

# Optimizer used for training model
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Retrieve the indices of the best 3 predictions
_, top_k_indices = tf.nn.top_k(y_conv, k=3)

# Compare each of the best 3 predictions with the actual labels
prediction_top_1 = tf.equal(top_k_indices[:, 0], tf.argmax(Y, 1, output_type=tf.int32))
prediction_top_2 = tf.equal(top_k_indices[:, 1], tf.argmax(Y, 1, output_type=tf.int32))
prediction_top_3 = tf.equal(top_k_indices[:, 2], tf.argmax(Y, 1, output_type=tf.int32))

# Calculate accuracies for top-1, top-2 and top-3 predictions
accuracy_top_1 = tf.reduce_mean(tf.cast(prediction_top_1, tf.float32))
accuracy_top_2 = tf.reduce_mean(tf.cast(tf.reduce_any(
    tf.stack([prediction_top_1, prediction_top_2]), axis=0), tf.float32))
accuracy_top_3 = tf.reduce_mean(tf.cast(tf.reduce_any(
    tf.stack([prediction_top_1, prediction_top_2, prediction_top_3]), axis=0), tf.float32))

# Store scalars for accuracy and loss
tf.summary.scalar('accuracy', accuracy_top_1)
tf.summary.scalar('loss', cost)

# Writers for storing tensorboard statistics
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("../../logs/train")
test_writer = tf.summary.FileWriter("../../logs/test")

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
    test_accuracies_top_1 = []
    test_accuracies_top_2 = []
    test_accuracies_top_3 = []
    test_losses = []

    for i in range(iterations):
        # Construct the batch
        if start == svhn.train_examples:
            start = 0
        end = min(svhn.train_examples, start + batch_size)

        batch_x = svhn.train_data[start:end]
        batch_y = svhn.train_labels[start:end]

        start = end

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if (i + 1) % display_step == 0 or i == 0:
            _accuracy_top_1, _accuracy_top_2, _accuracy_top_3, _cost, _summary = \
                sess.run([accuracy_top_1, accuracy_top_2, accuracy_top_3, cost, merged],
                         feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})

            print("Step: {0:6d}, "
                  "Training Accuracy (Top 1): {1:5f}, "
                  "Training Accuracy (Top 2): {2:5f}, "
                  "Training Accuracy (Top 3): {3:5f}, "
                  "Batch Loss: {4:5f}".format(i + 1, _accuracy_top_1, _accuracy_top_2, _accuracy_top_3, _cost))
            train_accuracies.append(_accuracy_top_1)
            train_losses.append(_cost)
            train_writer.add_summary(_summary, i)

    # Test the model by measuring it's accuracy
    test_iterations = svhn.test_examples // batch_size + 1
    for i in range(test_iterations):
        batch_x, batch_y = (svhn.test_data[i * batch_size:(i + 1) * batch_size],
                            svhn.test_labels[i * batch_size:(i + 1) * batch_size])
        _accuracy_top_1, _accuracy_top_2, _accuracy_top_3, _cost, _summary = \
            sess.run([accuracy_top_1, accuracy_top_2, accuracy_top_3, cost, merged],
                     feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        test_accuracies_top_1.append(_accuracy_top_1)
        test_accuracies_top_2.append(_accuracy_top_2)
        test_accuracies_top_3.append(_accuracy_top_3)
        test_losses.append(_cost)
        test_writer.add_summary(_summary, i)
    print("Mean Test Accuracy (Top 1): {0:5f}, "
          "Mean Test Accuracy (Top 2): {1:5f}, "
          "Mean Test Accuracy (Top 3): {2:5f}, "
          "Mean Test Loss: {3:5f}".format(np.mean(test_accuracies_top_1),
                                          np.mean(test_accuracies_top_2),
                                          np.mean(test_accuracies_top_3),
                                          np.mean(test_losses)))

    # print execution time
    print("Execution time in seconds: " + str(time.time() - start_time))
    train_writer.close()

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
    ax[1, 0].plot(range(0, test_iterations), test_accuracies_top_1, linewidth=1, color="darkgreen")

    # Test Loss
    ax[1, 1].set_title("Test Loss per Batch")
    ax[1, 1].set_xlabel("Batch")
    ax[1, 1].set_ylabel("Loss")
    ax[1, 1].set_ylim([0, max(test_losses)])
    ax[1, 1].plot(range(0, test_iterations), test_losses, linewidth=1, color="darkred")

    for i in range(1, iterations, svhn.train_examples // batch_size):
        ax[0, 0].axvline(x=i, ymin=0, ymax=1.05, linewidth=2, color="orange", label="skdlhv")
        ax[0, 1].axvline(x=i, ymin=0, ymax=max(train_losses), linewidth=2, color="orange")

    plt.show()

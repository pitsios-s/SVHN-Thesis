import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_multi_digit_labels(data_dir):
    digit_structure_file = data_dir + "/digitStruct.mat"
    file = h5py.File(digit_structure_file, 'r')
    digit_structure_bbox = file['digitStruct']['bbox']

    labels = []
    for i in range(len(digit_structure_bbox)):
        bb = digit_structure_bbox[i].item()

        attr = file[bb]["label"]
        if len(attr) > 1:
            label = [file[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            label = [attr.value[0][0]]

        labels.append(label)

    return labels


def calculate_digit_lengths(labels):
    lengths = {}
    for label in labels:
        length = len(label)
        if length not in lengths:
            lengths[length] = 1
        else:
            lengths[length] += 1

    return lengths


def main():
    train_labels_lengths = calculate_digit_lengths(load_multi_digit_labels("../../res/original/train"))
    test_labels_lengths = calculate_digit_lengths(load_multi_digit_labels("../../res/original/test"))
    extra_labels_lengths = calculate_digit_lengths(load_multi_digit_labels("../../res/original/extra"))

    # Visualize results
    plt.style.use("ggplot")
    train_indexes = np.arange(len(train_labels_lengths)) + 1
    test_indexes = np.arange(len(test_labels_lengths)) + 1
    extra_indexes = np.arange(len(extra_labels_lengths)) + 1
    fig, ax = plt.subplots(1, 3)

    # Bar chart for train labels
    ax[0].set_title("Digit Lengths for Train Dataset")
    train_bar = ax[0].bar(train_indexes, train_labels_lengths.values(), color="blue")
    ax[0].set_xticks(train_indexes)

    for bar in train_bar:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2, 1.01 * height, str(int(height)), ha="center", va="bottom")

    # Bar chart for test labels
    ax[1].set_title("Digit Lengths for Test Dataset")
    test_bar = ax[1].bar(test_indexes, test_labels_lengths.values(), color="blue")
    ax[1].set_xticks(test_indexes)

    for bar in test_bar:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width() / 2, 1.02 * height, str(int(height)), ha="center", va="bottom")

    # Bar chart for train labels
    ax[2].set_title("Digit Lengths for Extra Dataset")
    extra_bar = ax[2].bar(extra_indexes, extra_labels_lengths.values(), color="blue")
    ax[2].set_xticks(extra_indexes)

    for bar in extra_bar:
        height = bar.get_height()
        ax[2].text(bar.get_x() + bar.get_width() / 2, 1.02 * height, str(int(height)), ha="center", va="bottom")

    plt.show()


if __name__ == '__main__':
    main()

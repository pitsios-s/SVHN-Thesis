import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def calculate_frequencies(file_path):
    frequencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    labels = sio.loadmat(file_path)["y"]

    for label in labels:
        lab = label[0]
        if lab == 10:
            lab = 0
        frequencies[lab] += 1

    return frequencies


def main():
    train_labels_frequencies = calculate_frequencies("../../res/cropped/train_32x32.mat")
    test_labels_frequencies = calculate_frequencies("../../res/cropped/test_32x32.mat")
    extra_labels_frequencies = calculate_frequencies("../../res/cropped/extra_32x32.mat")

    # Visualize results
    plt.style.use("ggplot")
    indexes = np.arange(10)
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("Digit Occurrences for Train Dataset")
    ax[0].bar(indexes, train_labels_frequencies.values(), color="blue")
    ax[0].set_xticks(indexes)

    ax[1].set_title("Digit Occurrences for Test Dataset")
    ax[1].bar(indexes, test_labels_frequencies.values(), color="blue")
    ax[1].set_xticks(indexes)

    ax[2].set_title("Digit Occurrences for Extra Dataset")
    ax[2].bar(indexes, extra_labels_frequencies.values(), color="blue")
    ax[2].set_xticks(indexes)

    plt.show()


if __name__ == '__main__':
    main()

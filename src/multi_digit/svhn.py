import h5py
import os
import numpy as np
import PIL.Image as Image


class SVHNMulti:
    """ This class contains all the necessary functionality to Load, Process and Save the SVHN multi digit images """

    def __init__(self, output_dir, max_labels=5, normalize=True, gray=False):
        """ Default constructor
        Args:
            output_dir: The directory the processed data will be stored
            max_labels: The maximum number of digit length that we will allow
            normalize: Image normalization flag
            gray: Image gray scaling flag
        """
        self.PIXEL_DEPTH = 255
        self.NUM_LABELS = 11
        self.OUT_HEIGHT = 64
        self.OUT_WIDTH = 64
        self.NUM_CHANNELS = 3
        self.MAX_LABELS = max_labels
        self.output_dir = output_dir
        self.file = None
        self.digit_struct_name = None
        self.digit_struct_bbox = None
        self.normalize = normalize
        self.gray = gray

    def get_image_name(self, n):
        """Returns the current image's name
        Args:
            n: The index of the image

        Returns:
            The name of the n-th image
        """
        name = ''.join([chr(c[0]) for c in self.file[self.digit_struct_name[n][0]].value])
        return name

    def bounding_box_helper(self, attr):
        """Extracts and returns the value of a specific attribute of the bounding box structure

        Args:
            attr: The attribute to extract its value
        Returns:
            The value of the attribute
        """
        if len(attr) > 1:
            attr = [self.file[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def get_bounding_box(self, n):
        """Extracts and returns the bounding box information, regarding the n-th image

        Args:
            n: The index of the image

        Returns:
            Bounding box information about the n-th image
        """
        bbox = {}
        bb = self.digit_struct_bbox[n].item()

        bbox['label'] = self.bounding_box_helper(self.file[bb]["label"])
        bbox['top'] = self.bounding_box_helper(self.file[bb]["top"])
        bbox['left'] = self.bounding_box_helper(self.file[bb]["left"])
        bbox['height'] = self.bounding_box_helper(self.file[bb]["height"])
        bbox['width'] = self.bounding_box_helper(self.file[bb]["width"])

        return bbox

    def get_digit_structure(self, n):
        """Creates and returns the whole structure of the image, including the name, based on bounding boxes

        Args:
            n: The index of the image under consideration

        Returns:
            The image's structure
        """
        structure = self.get_bounding_box(n)
        structure['name'] = self.get_image_name(n)
        return structure

    def get_all_images_and_digit_structures(self):
        """Loops through every image and returns a large array containing the structure of each one"""
        structs = []
        for i in range(len(self.digit_struct_name)):
            structure = self.get_digit_structure(i)
            if len(structure["label"]) <= self.MAX_LABELS:
                structs.append(structure)
        return structs

    def save_data(self, data, labels, name):
        """Saves image data and labels in hdf5 format

        Args:
            data: The image data array
            labels: The labels of the images
            name: The name of the file to created
        """
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), "w")
        h5f.create_dataset(name + "_dataset", data=data)
        h5f.create_dataset(name + "_labels", data=labels)

    def load_data(self, name):
        """Loads an hdf5 file that contains the image data and labels

        Args:
            name: The name of the file to be loaded

        Returns:
            The data and labels arrays
        """
        h5f = h5py.File(os.path.join(self.output_dir, name + ".h5"), "r")
        data = h5f[name + "_dataset"][:]
        labels = h5f[name + "_labels"][:]

        return data, labels

    def read_digit_structure(self, data_dir):
        """Loads the digit structure file, stores the image name and bounding boxes details and returns an array
        containing the image name, as well as arrays containing the information about bounding boxes

        Args:
            data_dir: The directory that contains the digit structure file
        Returns:
            All the information from the structure file
        """
        struct_file = data_dir + "/digitStruct.mat"
        self.file = h5py.File(struct_file, 'r')
        self.digit_struct_name = self.file['digitStruct']['name']
        self.digit_struct_bbox = self.file['digitStruct']['bbox']
        structs = self.get_all_images_and_digit_structures()

        return structs

    def process_file(self, data_dir):
        """Processes all images one by one and returns them, together with their labels

        Args:
            data_dir: The directory that contains the images and the structure file

        Returns:
            The processed images and their labels
        """
        if self.gray:
            self.NUM_CHANNELS = 1

        structs = self.read_digit_structure(data_dir)
        data_count = len(structs)

        image_data = np.zeros((data_count, self.OUT_HEIGHT, self.OUT_WIDTH, self.NUM_CHANNELS), dtype=np.float32)
        labels = np.zeros((data_count, self.MAX_LABELS, self.NUM_LABELS), dtype=np.int32)

        for i in range(data_count):
            lbls = structs[i]['label']
            file_name = os.path.join(data_dir, structs[i]['name'])
            top = structs[i]['top']
            left = structs[i]['left']
            height = structs[i]['height']
            width = structs[i]['width']

            labels[i] = self.create_label_array(lbls)
            image_data[i] = self.create_image_array(file_name, top, left, height, width)

        return image_data, labels

    def create_label_array(self, labels):
        """Creates and returns an array of one-hot-encoded arrays, one for every digit in of the input label

        Args:
            labels: The labels of the current image, i.e a number with 1 to 5 digits

        Returns:
            An array of one-hot-encoded representations, for every digit in the input label
        """
        num_digits = len(labels)
        labels_array = np.ones([self.MAX_LABELS], dtype=np.int32) * 10
        one_hot_labels = np.zeros((self.MAX_LABELS, self.NUM_LABELS), dtype=np.int32)

        for n in range(num_digits):
            if labels[n] == 10:
                labels[n] = 0
            labels_array[n] = labels[n]

        for n in range(len(labels_array)):
            one_hot_labels[n] = self.one_hot_encode(labels_array[n])

        return one_hot_labels

    def one_hot_encode(self, number):
        """ Creates and returns a hot-hot-encoding representation of a given number
        Args:
            number: The number to be encoded
        Returns:
            The one-hot representation of the given number as a numpy array
        """
        one_hot = np.zeros(shape=self.NUM_LABELS, dtype=np.int32)
        one_hot[number] = 1

        return one_hot

    def create_image_array(self, file_name, top, left, height, width):
        """Crops an image so that it will contain only the portion with the digits of interest

        Args:
            file_name: The name of the image to be processed
            top: The Y-coordinate of the top left point of every digit's bounding box
            left: The X-coordinate of the top left point of every digit's bounding box
            height: The height of every digit's bounding box
            width: The width of every digit's bounding box

        Returns:
            A numpy representation of the processed image
        """

        # Load image
        image = Image.open(file_name)

        # Find the smallest Y-coordinate among every digit's bounding box
        image_top = np.amin(top)

        # Find the smallest X-coordinate among every digit's bounding box
        image_left = np.amin(left)
        image_height = np.amax(top) + height[np.argmax(top)] - image_top
        image_width = np.amax(left) + width[np.argmax(left)] - image_left

        # Find the smallest possible bounding box that will fit all the digits an once, and expand it by 20%
        box_left = np.floor(image_left - 0.1 * image_width)
        box_top = np.floor(image_top - 0.1 * image_height)
        box_right = np.amin([np.ceil(box_left + 1.2 * image_width), image.size[0]])
        box_bottom = np.amin([np.ceil(image_top + 1.2 * image_height), image.size[1]])

        # Crop image based on the unified bounding box and scale it to a smaller size
        image = image.crop((box_left, box_top, box_right, box_bottom)).\
            resize([self.OUT_HEIGHT, self.OUT_WIDTH], Image.ANTIALIAS)

        # Convert image to numpy array
        image_array = np.array(image)

        # Normalize if necessary
        if self.normalize:
            image_array = (255 - image_array) * 1.0 / 255.0
            image_array -= np.mean(image_array, axis=0)

        # Convert to gray scale if necessary
        if self.gray:
            image_array = np.expand_dims(np.dot(image_array, [0.2989, 0.5870, 0.1140]), axis=3)

        return image_array


def main():
    svhn = SVHNMulti("../../res/processed/normalized", max_labels=2, normalize=True, gray=False)

    # Train dataset
    train_data, train_labels = svhn.process_file("../../res/original/train")
    svhn.save_data(train_data, train_labels, "train")

    # Test dataset
    test_data, test_labels = svhn.process_file("../../res/original/test")
    svhn.save_data(test_data, test_labels, "test")

    # Extra dataset
    extra_data, extra_labels = svhn.process_file("../../res/original/extra")
    svhn.save_data(extra_data, extra_labels, "extra")


if __name__ == '__main__':
    main()

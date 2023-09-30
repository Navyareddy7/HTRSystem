"""The provided code defines a DataLoaderIAM class that is responsible for loading data in the IAM dataset format for handwriting recognition tasks. The IAM dataset is a popular dataset used for optical character recognition (OCR) research.

Here's an explanation of the main parts of the code:

Importing necessary modules and defining namedtuples:

pickle: For serializing and deserializing Python objects.
random: For randomizing the dataset during training.
collections.namedtuple: For creating namedtuples, which are lightweight data structures used to hold the samples and batches of data.
cv2: For image processing tasks using OpenCV.
lmdb: For working with the LMDB database (used for fast data loading).
numpy: For numerical operations with arrays.
path.Path: A class from the path module, used for handling file paths.
Defining namedtuples:

Sample: A namedtuple that represents a single data sample. It contains two fields: gt_text (ground truth text) and file_path (the path to the image file).
Batch: A namedtuple that represents a batch of data. It contains three fields: imgs (a list of images in the batch), gt_texts (a list of corresponding ground truth texts), and batch_size (the size of the batch).
The DataLoaderIAM class:

The class constructor (__init__) initializes the data loader with the necessary parameters:

data_dir: The path to the dataset directory.
batch_size: The desired batch size for training or validation.
data_split: The fraction of data to be used for training (default is 0.95, meaning 95% for training and 5% for validation).
fast: A boolean flag to enable fast data loading using LMDB (default is True).
The class has methods for managing the dataset:

train_set: Switches to the training set, shuffles the samples, and enables data augmentation.
validation_set: Switches to the validation set and disables data augmentation.
get_iterator_info: Returns the current batch index and the total number of batches.
has_next: Checks if there is a next element (i.e., if there is more data to be processed).
_get_img: A private method to load an image from the dataset. It uses LMDB if the fast flag is set, or else it uses OpenCV to read the image from disk.
The core method for getting the next batch of data is get_next, which returns a Batch namedtuple containing a batch of images and their corresponding ground truth texts.

Dataset Preparation:

The __init__ method reads the dataset and loads the ground truth texts and image file paths from the dataset's 'words.txt' file.
It splits the data into training and validation sets based on the provided data_split.
The list of all unique characters (chars) present in the dataset is created.
Data Loading:

During training, data augmentation is enabled (self.data_augmentation = True), and the data is shuffled to increase randomness.
During validation, data augmentation is disabled (self.data_augmentation = False), and the data is not shuffled.
Data Retrieval:

The get_next method loads the next batch of data, including images and their ground truth texts, by iterating through the dataset.
This code provides the necessary functionality to load and manage data for handwriting recognition tasks using the IAM dataset."""






import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

   
 
    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        for line in f:
            # ignore empty and comment lines
            line = line.strip()
            if not line or line[0] == '#':
                continue

            line_split = line.split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = line_split[0] + '.png'
            file_name = data_dir / 'img' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            # GT text are columns starting at 9
            gt_text = ' '.join(line_split[8:])
            chars = chars.union(set(list(gt_text)))

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # split into training and validation set: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

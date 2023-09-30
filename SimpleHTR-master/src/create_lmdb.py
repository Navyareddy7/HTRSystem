import argparse  # For parsing command-line arguments.



import pickle #pickle: For serializing and deserializing Python objects.

import cv2 #cv2 (OpenCV): For image processing tasks like reading the grayscale PNG images.
import lmdb #lmdb: For working with the LMDB database.
from path import Path #Path from the path module: This is used for handling file paths in a platform-independent way.




""" Here, an argument parser is created using argparse.ArgumentParser(). It is set up to accept a single argument named --data_dir. The type=Path argument specifies that the value provided for --data_dir will be converted to a Path object. The required=True argument ensures that the --data_dir argument must be provided when running the script."""
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

"""This line checks whether an LMDB directory already exists in the specified --data_dir. It uses the exists() method of the Path object to check if the directory exists. If the directory exists, the assert statement will raise an AssertionError, stopping the script execution to prevent overwriting existing data."""
# 2GB is enough for IAM dataset
assert not (args.data_dir / 'lmdb').exists()
"""This line opens an LMDB environment with a maximum map size of 2GB. The LMDB environment is created in the directory --data_dir/lmdb. The str() function is used to convert the Path object to a string representing the path to the LMDB directory."""
env = lmdb.open(str(args.data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 2)

# go over all png files

"""This line creates a list fn_imgs containing the file paths of all PNG image files found in the --data_dir/img subdirectory. The walkfiles('*.png') method from the path module is used to recursively search for all files with a .png extension in the subdirectory."""
fn_imgs = list((args.data_dir / 'img').walkfiles('*.png'))

# and put the imgs into lmdb as pickled grayscale imgs

"""Here, a loop iterates over each PNG image file found in the previous step. The loop uses an LMDB transaction (txn) with write access.
Inside the loop, it first prints the current iteration index and the total number of images (i and len(fn_imgs)).
The image is read using cv2.imread with the flag cv2.IMREAD_GRAYSCALE, which reads the image as a grayscale image.
The image's basename (filename without the directory) is extracted using fn_img.basename().
The image is serialized using pickle.dumps.
The serialized image is stored in the LMDB database using the basename as the key and the serialized image data as the value."""
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("ascii"), pickle.dumps(img))

env.close()

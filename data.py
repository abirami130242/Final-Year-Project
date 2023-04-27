import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

H = 512
W = 512

# RETURNS LIST OF FULL FILE PATHS FOR IMAGES AND CORRESPONDING MASKS
def process_data():
    folder = os.path.dirname(os.path.abspath(__file__))
    
    image_path = os.path.join(folder, "images")                                    ##########
    image_filenames = os.listdir(image_path)
    image_fullnames = []
    for count in range(len(image_filenames)):
        image_fullnames.append(os.path.join(image_path, image_filenames[count]))
    
    mask_path = os.path.join(folder, "masks")                                      ##########
    mask_filenames = os.listdir(mask_path)
    mask_fullnames = []
    for count in range(len(mask_filenames)):
        mask_fullnames.append(os.path.join(mask_path, mask_filenames[count]))
    
    return image_fullnames, mask_fullnames

# SPLITS LIST OF FULL FILE PATHS INTO TRAIN, VALID AND TEST LISTS
def load_data():
    images, masks = process_data()

    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)
    
    train_x, valid_x = train_test_split(train_x, test_size=0.25, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.25, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# READS IMAGE AS RGB (512, 512, 3), VALUES ARE 0.0 - 1.0
def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

# READS IMAGE AS GRAYSCALE (512, 512), PIXEL VALUES ARE 0 - 2 (background, choroid, sclera)
def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x[x == 255] = 2
    x[x == 128] = 1
    x[x == 0] = 0
    x = x.astype(np.int32)
    return x

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])

    return image, mask


if __name__ == "__main__":
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
    
    dataset = tf_dataset(train_x, train_y, batch=8)
    for x, y in dataset:
        print(x.shape, y.shape) ## (8, 512, 512, 3), (8, 512, 512, 3)

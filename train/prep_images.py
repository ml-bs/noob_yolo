import tensorflow as tf
import numpy as np
import os
from random import random

def resize(url, out_dir, out_name, i):
    image = tf.io.read_file(url)
    image = tf.io.decode_image(image, dtype = tf.float32)
    image = tf.image.resize_with_pad(image, 416, 416)

    if i%7 == 0:
        image = tf.image.rot90(image)
    elif i%7 == 1:
        image = tf.image.rot90(image, 2)
    elif i%7 == 2:
        image = tf.image.rot90(image, 3)

    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.io.encode_jpeg(image, name = out_name)

    out = os.path.join(out_dir, out_name + ".jpeg")

    tf.io.write_file(out, image)

def augment(url, out_dir, out_name):
    image = tf.io.read_file(url)
    image = tf.io.decode_image(image, dtype = tf.float32)

    gray_image = tf.image.rgb_to_grayscale(image)
    gray_image = tf.image.convert_image_dtype(gray_image, tf.uint8)
    gray_image = tf.io.encode_jpeg(gray_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_gray.jpeg")
    tf.io.write_file(out, gray_image)

    bright_image = tf.image.adjust_brightness(image, random() * 0.5)
    bright_image = tf.image.convert_image_dtype(bright_image, tf.uint8)
    bright_image = tf.io.encode_jpeg(bright_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_bright.jpeg")
    tf.io.write_file(out, bright_image)

    contrast_image = tf.image.adjust_contrast(image, random() * 2)
    contrast_image = tf.image.convert_image_dtype(contrast_image, tf.uint8)
    contrast_image = tf.io.encode_jpeg(contrast_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_contrast.jpeg")
    tf.io.write_file(out, contrast_image)

    hue_image = tf.image.adjust_hue(image, random() - 0.5)
    hue_image = tf.image.convert_image_dtype(hue_image, tf.uint8)
    hue_image = tf.io.encode_jpeg(hue_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_hue.jpeg")
    tf.io.write_file(out, hue_image)

    saturated_image = tf.image.adjust_saturation(image, random() * 8)
    saturated_image = tf.image.convert_image_dtype(saturated_image, tf.uint8)
    saturated_image = tf.io.encode_jpeg(saturated_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_saturated.jpeg")
    tf.io.write_file(out, saturated_image)

    noise = tf.random.normal(shape = tf.shape(image), mean = 0.0, stddev = 0.2, dtype = tf.float32)
    noise_image = image + noise
    noise_image = tf.image.convert_image_dtype(noise_image, tf.uint8)
    noise_image = tf.io.encode_jpeg(noise_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_noise.jpeg")
    tf.io.write_file(out, noise_image)

source_dir = "../data/rowdy"

target_dir = "../data/doggo_pics"

#image prepping
"""
i = 0
for filename in os.listdir(source_dir):
    url = os.path.join(source_dir, filename)
    resize(url, target_dir, str(i), i)
    i += 1
"""

def process_image(image_url):
    image = tf.io.read_file(image_url)
    image = tf.io.decode_image(image, dtype = tf.float32)
    return image

def load_dataset():

    images_dir = "../data/doggo_pics"
    labels_dir = "../data/doggo_labels"

    images = []
    labels = []

    for image_filename in os.listdir(images_dir):
        image_url = os.path.join(images_dir, image_filename)
        
        label_filename = image_filename.replace(".jpeg", ".txt")
        label_url = os.path.join(labels_dir, label_filename)
        
        label = open(label_url)
        label = np.genfromtxt(label, delimiter = " ")
        label = tf.cast(label, tf.float32)

        print(label)

        images.append(image_url)
        labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y : (
                process_image(x),
                y
                )
            )

    return dataset

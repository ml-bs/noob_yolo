import tensorflow as tf
import os
from random import random
import numpy as np
from skimage.util import random_noise

def resize(url, out_dir, out_name):
    image = tf.io.read_file(url)
    image = tf.io.decode_image(image, dtype = tf.float32)
    image = tf.image.resize_with_pad(image, 400, 400)

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

    contrast_image = tf.image.adjust_contrast(image, random() * 0.7)
    contrast_image = tf.image.convert_image_dtype(contrast_image, tf.uint8)
    contrast_image = tf.io.encode_jpeg(contrast_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_contrast.jpeg")
    tf.io.write_file(out, contrast_image)

    hue_image = tf.image.adjust_hue(image, random() - 0.5)
    hue_image = tf.image.convert_image_dtype(hue_image, tf.uint8)
    hue_image = tf.io.encode_jpeg(hue_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_hue.jpeg")
    tf.io.write_file(out, hue_image)

    saturated_image = tf.image.adjust_saturation(image, random() * 7)
    saturated_image = tf.image.convert_image_dtype(saturated_image, tf.uint8)
    saturated_image = tf.io.encode_jpeg(saturated_image, name = out_name)
    out = os.path.join(out_dir, out_name + "_saturated.jpeg")
    tf.io.write_file(out, saturated_image)

#   noise_image = random_noise(image, mode = "gaussian", var = random() * 0.03)
#   image = tf.image.convert_image_dtype(image, tf.uint8)
#    out = os.path.join(out_dir, out_name + "_noise.jpeg")
#    tf.io.write_file(out, image)

url = "../data/rowdy/2238.jpeg"
augment(url, "./", "temp")

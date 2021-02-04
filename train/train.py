import tensorflow as tf
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from yolov3.models import (YoloSmol, YoloLoss, smol_anchors, smol_anchor_masks)
from misc.data_load_preprocess import (process_target, target_to_out_format, image_normalize)
from misc.constants import constants
from prep_images import load_dataset

def main():
    model = YoloSmol(constants["size"], training = True, classes = constants["num_classes"])
    anchors = smol_anchors
    anchor_masks = smol_anchor_masks

    train_dataset = load_dataset() 

    train_dataset = train_dataset.shuffle(80)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.map(lambda x, y : (
            image_normalize(x),
            process_target(y, anchors, anchor_masks, constants["size"])
        ))

    optimizer = tf.keras.optimizers.Adam()
    loss = [YoloLoss(anchors[mask], constants["num_classes"]) for mask in anchor_masks]

    model.compile(optimizer = optimizer, loss = loss)

    model.fit(train_dataset, epochs = constants["epochs"], validation_split = 0.3)

main()

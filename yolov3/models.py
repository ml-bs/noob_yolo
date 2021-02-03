import tensorflow as tf
import numpy as np

from .utils import iou
from misc.constants import constants

smol_anchors = np.array([(), (), (), (), (), ()]) #decide on the anchor values

smol_anchor_masks = np.array([[0, 1, 2], [3, 4, 5]]) #first term represents the smaller anchors

#convolutional layer with batch normalization
def DarknetConv(x, filters, kernel_size, strides = 1):
    
    #apply padding depending on strides
    if strides == 1:
        padding = "same"
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"

    #Conv2D layer
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = False, kernel_regularizer = tf.keras.regularizers.L2(0.0005))(x)

    #apply batch normalization + leaky relu(activation)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.1)(x)

    return x

#may be unnecessary
#residual layer, save copy of x, perform convolution, add current x and saved copy to implement skip forward
"""
def DarknetRes(x, filters):
    prev = x

    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)

    x = tf.keras.layers.Add()([prev, x])

    return x
"""

#normal sized darknet (may take very long to train)
#def Darknet(name="placeholder"):

#small darknet for faster training
def DarknetSmol(name="placeholder"):
    x = inputs = tf.keras.layers.Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, padding = "same")(x)
    x = DarknetConv(x, 32, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, padding = "same")(x)
    x = DarknetConv(x, 64, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, padding = "same")(x)
    x = DarknetConv(x, 128, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, padding = "same")(x)
    x = x_skip = DarknetConv(x, 256, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, padding = "same")(x)
    x = DarknetConv(x, 512, 3)
    x = tf.keras.layers.MaxPool2D(2, 1, padding = "same")(x)
    x = DarknetConv(x, 1024, 3)

    return tf.keras.Model(inputs, (x_skip, x), name = name)

#normal yolo convolutional layers (layers that come after darknet)
#def YoloConv():

#small yolo convolutional layers
def YoloConvSmol(x_in, filters, name = "placeholder"):
    if isinstance(x_in, tuple):
        inputs = tf.keras.layers.Input(x_in[0].shape[1:]), tf.keras.layers.Input(x_in[1].shape[1:])
        x, x_skip = inputs
        x = DarknetConv(x, filters, 1)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, x_skip])

    else:
        x = inputs = tf.keras.layers.Input(x_in.shape[1:])
        x = DarknetConv(x, filters, 1)

    return tf.keras.Model(inputs, x, name = name)(x_in)

#yolo output layer, final layer with 2 darknet convs, returns output in the desired shape
#output is to be returned as a 13x13 or 26x26 grid (for the smaller object detection)
#each grid square contains information about 3 boxes
#each box is represented as a set of values - 4 position values - x, y, width, height, 1 objectness value - the probability of there being an object in the box,
#and some class_probs values, depending on the number of classes - indicating the probs of the object belonging to each class
def YoloOut(x_in, filters, num_anchors, num_classes, name = "placeholder"):
    x = inputs = tf.keras.layers.Input(x_in.shape[1:])
    x = DarknetConv(x, filters * 2, 1)
    x = tf.keras.layers.Conv2D(filters = num_anchors * (num_classes + 5), kernel_size = 1, strides = 1, padding = "same", use_bias = True, kernel_regularizer = tf.keras.regularizers.L2(0.0005))(x)
    x = tf.keras.layers.Lambda(lambda x : tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors, num_classes + 5)))(x)

    return tf.keras.Model(inputs, x, name = name)(x_in)

#generate bounding boxes, extract information from given raw predictions
#pred is of the following format - some number n of 13 x 13 or 26 x 26 grids, each grid box containing info for 3 boxes in the format (x, y, w, h, objectness, class_probs) 
def yolo_boxes(pred, anchors, num_classes):
    grid_size = tf.shape(pred)[1:3]
    #axis = -1 splits it along the last axis, which in this case is the pred value containing the box information
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, num_classes), axis = -1) 

    #at this point, box_xy is a 13x13 or 26x26 grid, each grid box containing box positions for 3 boxes
    #box_wh similarly contains box extents, objectness contains object probs for each box, class_probs contains  class_probs

    #also note - the positions an extents at this point are relative to the top left corner of the grid cell they fall in

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis = -1) #raw box position and extents for use in loss function

    #now process the raw preds to obtain usable bounding boxes
    #also transform box positions and extents from cell relative to absolute

    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis = -1), axis = 2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    #bounding boxes can be defined by their top left and bottom right points
    box_xy_topleft= box_xy - box_wh / 2
    box_xy_bottomright = box_xy + box_wh / 2
    bounding_box = tf.concat([box_xy_topleft, box_xy_bottomright], axis = -1)

    return bounding_box, objectness, class_probs, pred_box

#perform non max suppression on given boxes
def nms(box_outputs):
    boxes_temp = []
    confidence_temp = []
    class_probs_temp = []

    #at this point, the bounding boxes in box_outputs are in the shape (batch_size, 13, 13, 4) (4 dimensions)
    #we consider all boxes equally when performing nms, so we reshape it to (batch_size, 13*13, 4) (3 dimensions)
    #similarly for confidence (objectness) and class_probs

    for o in box_outputs:
        boxes_temp.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        confidence_temp.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        class_probs_temp.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bounding_boxes = tf.concat(boxes_temp, axis = -1)
    confidence = tf.concat(confidence_temp, axis = -1)
    class_probs = tf.concat(class_probs_temp, axis = -1)

    #the final score for a box is formed by - 
    scores  = confidence * class_probs

    #performing nms
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(bounding_boxes, (tf.shape(bounding_boxes)[0], -1, 1, 4)), 
            scores = tf.reshape(scores, (tf.shape(scores[0], -1, tf.shape(scores)[-1]))),
            max_output_size_per_class = constants["max_boxes"],
            max_total_size = constants["max_boxes"],
            iou_threshold = constants["iou_threshold"],
            score_threshold = constants["score_threshold"]
            )

    return boxes, scores, classes, valid_detections

    
#normal sized yolo network (darknet + yoloconv + yoloout + yoloboxes + nms) 
#def Yolo():

#small yolo network (darknetsmol + yoloconvsmol + yoloout + yoloboxes + nms)
def YoloSmol(size = None, channels = 3, anchors = smol_anchors, masks = smol_anchor_masks, classes = constants["num_classes"], training = False):
    x = inputs = tf.keras.layers.Input([size, size, channels], name = "input")

    #applying the base darknet
    x_skip, x = DarknetSmol(name = "darknet")(x)

    x = YoloConvSmol(x_in = x, filters = 256, name = "yolo_conv_full")
    outputs_0 = YoloOut(x_in = x, filters = 256, num_anchors = len(masks[1]), num_classes = classes, name = "out_from_full")

    x = YoloConvSmol(x_in = (x, x_skip), filters = 128, name = "yolo_conv_skip")
    outputs_1 = YoloOut(x_in = x, filters = 128, num_anchors = len(masks[0]), num_classes = classes, name = "out_from_skip")

    #if training, return raw pred values
    if training:
        return tf.keras.Model(inputs, (outputs_0, outputs_1), name = "yolo_training")
    
    boxes_0 = tf.keras.layers.Lambda(lambda x : yolo_boxes(x, anchors[masks[1]], classes), name = "yolo_boxes_0")(outputs_0)
    boxes_1 = tf.keras.layers.Lambda(lambda x : yolo_boxes(x, anchors[masks[0]], classes), name = "yolo_boxes_1")(outputs_1)

    outputs = tf.keras.layers.Lambda(lambda x : nms(x), name = "nms_final")((boxes_0[:3], boxes_1[:3]))

    return tf.keras.Model(inputs, outputs, name = "yolo")

#yolo loss function
def YoloLoss(anchors, classes = constants["num_classes"], ignore_thresh = 0.5):
    def yolo_loss(y, y_preds):
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_preds, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_class_idx = tf.split(y, (4, 1, 1), axis = -1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        #here, pred boxes are in their grid cell relative format, while true boaxes are in the normal absolute format
        #so we apply the inverse of the function applied in yolo_boxes to the true boxes, to make them comparable
        grid_size = tf.shape(y)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis = -1), axis = 2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid_size, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        #create obj mask to indicate which boxes contain an object
        obj_mask = tf.squeeze(true_obj, -1)
        best_iou = tf.map_fn(lambda x : tf.reduce_max(iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis = -1), (pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        #calculate losses, then sum
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis = -1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis = -1)
        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss #need to consider both, confidence in predicting object and in predicting no object, and penalize accordingly
        class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_class_idx, pred_class)

        xy_loss_total = tf.reduce_sum(xy_loss, axis = (1, 2, 3))
        wh_loss_total = tf.reduce_sum(wh_loss, axis = (1, 2, 3))
        obj_loss_total = tf.reduce_sum(obj_loss, axis = (1, 2, 3))
        class_loss_total = tf.reduce_sum(class_loss, axis = (1, 2, 3))

        return xy_loss_total + wh_loss_total + obj_loss_total + class_loss_total

    return yolo_loss

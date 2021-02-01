import tensorflow as tf
import numpy as np

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
    x = tf.keras.layers.Conv2D(filters = anchors * (classes + 5), kernel_size = kernel_size, strides = strides, padding = padding, use_bias = False, kernel_regularizer = tf.keras.regularizers.L2(0.0005))(x)

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
#each grid contains information about 3 boxes
#each box is represented as a set of values - 4 position values - x, y, width, height, 1 objectness value - the probability of there being an object in the box,
#and some class_probs values, depending on the number of classes - indicating the probs of the object belonging to each class
def YoloOut(x_in, filters, num_anchors, num_classes, name = "placeholder"):
    x =inputs = tf.keras.layers.Input(x_in.shape[1:])
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

    #at this point, box_xy is some number n of 13x13 or 26x26 grids, each grid box containing box positions for 3 boxes
    #box_wh similarly contains box extents, objectness contains object probs for each box, class_probs contains  class_probs

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis = -1) #raw box position and extents for use in loss function

    #now process the raw preds to obtain usable bounding boxes

    #create two grids of the given grid size, one to represent x coordiantes and one y
    #each grid cell contains two numbers from 0 to grid_size, representing its x and y indices
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis = -1), axis = 2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    #bounding boxes can be defined by their top left and bottom right points
    box_xy_topleft= box_xy - box_wh / 2
    box_xy_bottomright = box_xy + box_wh / 2
    bounding_box = tf.concat([box_xy_topleft, box_xy_bottomright], axis = -1)

    return bounding_box, objectness, class_probs, pred_box

#perform non max suppression on given boxes
def nms():
    
#normal sized yolo network (darknet + yoloconv + yoloout + yoloboxes + nms) 
#def Yolo():

#small yolo network (darknetsmol + yoloconvsmol + yoloout + yoloboxes + nms)
def YoloSmol():

#yolo loss function
def yolo_loss():

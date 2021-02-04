import tensorflow as tf

'''
@tf.function
def target_to_out_format(y_true, grid_size, anchor_ids):
    n = tf.shape(y_true)[0]
    
    y_true_out = tf.zeros((n, grid_size, grid_size, tf.shape(anchor_ids)[0], 6))

    anchor_ids = tf.cast(anchor_ids, tf.int32)
    
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size = True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size = True)
    idx = 0

    for i in tf.range(n):
        for j in tf.range(tf.shape(y_true)[1]):

            if tf.equal(y_true[i][j][3], 0):
                continue
            
            anchor_equality = tf.equal(anchor_ids, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_equality):
                box = y_true[i][j][1:5]
                box_xy = (y_true[i][j][1:3] + y_true[i][j][3:5]) / 2

                anchor_id = tf.cast(tf.where(anchor_equality), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_id[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][0]])

                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())

def process_target(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    anchors = tf.cast(anchors, tf.float32)
    anchor_areas = anchors[..., 0] * anchors[..., 1]

    box_wh = y_train[..., 3:5] - y_train[..., 1:3]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, tf.shape(anchors)[0], 1))

    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])

    iou = intersection / (box_area + anchor_areas - intersection)
    anchor_id = tf.cast(tf.argmax(iou, axis = -1), tf.float32)
    anchor_id = tf.expand_dims(anchor_id, axis = -1)

    y_train = tf.concat([y_train, anchor_id], axis = -1)

    for anchor_ids in anchor_masks:
        y_outs.append(target_to_out_format(y_train, grid_size, anchor_ids))
        grid_size *= 2

    return tuple(y_outs)

def image_normalize(x):
    return x / 255

'''

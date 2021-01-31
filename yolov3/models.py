import tensorflow as tf

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
def DarknetRes(x, filters):
    prev = x

    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)

    x = tf.keras.layers.Add()([prev, x])

    return x

def DarknetSmol(name="placeholder"):

def YoloConv():
    
def YoloSmol()

import keras, os
from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, LeakyReLU
from keras.layers.convolutional import Convolution2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from .. import config

"""
Generating model.
"""

model_checkpoint_dir = os.path.join(config.checkpoint_path(), "model.h5")
saved_model_dir = os.path.join(config.output_path(), config.test_model_name + ".h5")

"""
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu:
        x  = LeakyReLU(alpha=config.alpha_lrelu)(x)
    else:
        x = Activation('relu')(x)
    # -----------------------------------------------------------------------------
"""

def expand_conv(init, base, k, stride):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    layers = 0
#=====================================================================================================
    shortcut  = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu:
        shortcut  = LeakyReLU(alpha=config.alpha_lrelu)(shortcut)
    else:
        shortcut  = Activation('relu')(shortcut)
    # -----------------------------------------------------------------------------
    
    x = ZeroPadding2D((1, 1))(shortcut)
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
    layers += 4
#=====================================================================================================
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu:
        x  = LeakyReLU(alpha=config.alpha_lrelu)(x)
    else:
        x = Activation('relu')(x)
    # -----------------------------------------------------------------------------
    
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
    layers += 4
#=====================================================================================================
    # Add shortcut
    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)

    m = Add()([x, shortcut])
    layers += 2
    return [m, layers]


def conv_block(input, n, stride, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    layers = 0
#=====================================================================================================
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu:
        x  = LeakyReLU(alpha=config.alpha_lrelu)(x)
    else:
        x = Activation('relu')(x)
    # -----------------------------------------------------------------------------
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    layers += 3
#=====================================================================================================
    if dropout > 0.0:
        x = Dropout(dropout)(x)
        layers += 1
#=====================================================================================================
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu:
        x  = LeakyReLU(alpha=config.alpha_lrelu)(x)
    else:
        x = Activation('relu')(x)
    # -----------------------------------------------------------------------------
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    layers += 3
#=====================================================================================================
    m = Add()([init, x])
    layers += 1
    return [m, layers]

def create_wide_residual_network(input_dim, nb_classes=10, if_n=[1,1,1], N=2, k=1):
    """
    Creates a Wide Residual Network with specified parameters
    :input: Input shape of object
    :nb_classes: Number of output classes
    :N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :k: Width of the network.
    :dropout: Adds dropout if value is greater than 0.0
    """
    conv_2D = [16, 32, 64] # [16, 32, 64]
    dropout = [0.20, 0.25, 0.30] # [0.25, 0.25, 0.25]

    ip = Input(shape=input_dim)
    x = ZeroPadding2D((1, 1))(ip)

    
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    layers = 3
    nb_conv = 1
#=====================================================================================================
    if if_n[0]:
        x, l = expand_conv(x, conv_2D[0], k, stride=(1,1))
        layers += l
        nb_conv += 3
        for i in range(N - 1):
            x, l = conv_block(x, n=conv_2D[0], stride=(1,1), k=k, dropout=dropout[0])
            nb_conv += 2
            layers += l
#=====================================================================================================
    if if_n[1]:
        x, l = expand_conv(x, conv_2D[1], k, stride=(1,1))
        layers += l
        nb_conv += 3
        for i in range(N - 1):
            x, l = conv_block(x, n=conv_2D[1], stride=(2,2), k=k, dropout=dropout[1])
            nb_conv += 2
            layers += l
#=====================================================================================================
    if if_n[2]:
        x, l = expand_conv(x, conv_2D[2], k, stride=(1,1))
        layers += l
        nb_conv += 3
        for i in range(N - 1):
            x, l = conv_block(x, n=conv_2D[2], stride=(2,2), k=k, dropout=dropout[2])
            nb_conv += 2
            layers += l
#=====================================================================================================
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)
    layers += 3

    model = Model(ip, x)
    return [model, nb_conv, layers, conv_2D, dropout, N, k]

#read mode
def read_model():
    model = load_model(saved_model_dir)
    return model

#save check-point model
def saved_model_checkpoint():
    return ModelCheckpoint(model_checkpoint_dir, 
                            monitor=config.monitor, 
                            verbose=2, 
                            save_best_only=True, 
                            save_weights_only=False, 
                            mode='auto', 
                            period=1)

#early stopping of model
def set_early_stopping():
    return EarlyStopping(monitor=config.monitor,
                            patience= 15,
                            verbose=2,
                            mode='auto')

# learning_rate
#model_optimizer = Adam(lr=0.07, decay=1e-6) # Adam = momenterm + rmsprop
#model_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # base
model_optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
def reduce_lr():
    print("Optimizer :","Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)")
    return ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=5,
                            min_lr=0.001)
def get_model():
    init = (32, 32, 3)
    model, nb_conv, layers, conv_2D, dropout, N, k = create_wide_residual_network(init,nb_classes=10,
                                                    if_n=[0,1,0], N=1, k=6)
    model.summary()
    print("nb_conv block :", nb_conv)
    print("Total layers  :", layers)
    print("conv2D shape  :", conv_2D)
    print("Dropout shape :", dropout)
    print("N , k = ", N,',', k)
    return model

if __name__ == "__main__":
    init = (32, 32, 3)
    model, nb_conv, layers, conv_2D, dropout, N, k = create_wide_residual_network(init,nb_classes=10,
                                                    if_n=[0,1,0], N=1, k=6)
    model.summary()
    print("nb_conv block :", nb_conv)
    print("Total layers  :", layers)
    print("conv2D shape  :", conv_2D)
    print("Dropout shape :", dropout)
    print("N , k = ", N,',', k)

print("\n\n=========================================")
print("Model generate successfully.")
print("=========================================\n\n")

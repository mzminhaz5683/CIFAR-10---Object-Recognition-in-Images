import keras, os
from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, LeakyReLU
from keras.layers.convolutional import Convolution2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.merge import concatenate

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from .. import config

"""
Generating model.
"""
init        = (32, 32, 3)
nb_classes  = 10
if_n        = [1,1,1]
loop        = 1
pool        = 2
gnet        = 0
conv_2D     = [16, 32, 64] # [16, 32, 64]
dropout     = [0.20, 0.25, 0.30] # [0.25, 0.25, 0.25]
#==================================================================================================
"""
    # -----------------------------------------------------------------------------
    if config.alpha_lrelu_lrelu:
        x  = LeakyReLU(alpha=config.alpha_lrelu_lrelu)(x)
    else:
        x = Activation('relu')(x)
    # -----------------------------------------------------------------------------
"""


model_checkpoint_dir = os.path.join(config.checkpoint_path(), "model.h5")
saved_model_dir = os.path.join(config.output_path(), config.test_model_name + ".h5")
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
# function for creating a naive inception block
def googlenet_block(x):
    if gnet:
        """                    _______
            (1, 1)-------------|      |
            (1, 1) -> (3, 3)---| con- |_____
            (1, 1) -> (5, 5)---| cate |
            pool() -> (1, 1)---|______|
        """
        # 1x1 conv
        conv1 = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        # 3x3 conv
        conv3_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        conv3 = Convolution2D(conv_2D[1], (3,3), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(conv3_conv)
        # 5x5 conv
        conv5_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        conv5 = Convolution2D(conv_2D[2], (5,5), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(conv5_conv)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        pool_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(pool)
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool_conv])
        
        layers = 8
        conv_n = 6
        return [layer_out, layers, conv_n]
    else:
        """                    _______
            (1, 1)-------------|      |
            (3, 3) -> (1, 1)---| con- |_____
            (5, 5) -> (1, 1)---| cate |
            pool() -> (1, 1)---|______|
        """
        # 1x1 conv
        conv1 = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        # 3x3 conv
        conv3 = Convolution2D(conv_2D[1], (3,3), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        conv3_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(conv3)
        # 5x5 conv
        conv5 = Convolution2D(conv_2D[2], (5,5), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(x)
        conv5_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(conv5)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        pool_conv = Convolution2D(conv_2D[0], (1,1), padding='same', activation='relu',
                                                                kernel_initializer='he_normal', use_bias=False)(pool)
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3_conv, conv5_conv, pool_conv])

        layers = 8
        conv_n = 6
        return [layer_out, layers, conv_n]
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
def resnet_block(input, dropout_n):
    layers = conv_n = 0
    init_r, l, c = googlenet_block(input)
    layers += l
    conv_n += c
#=====================================================================================================
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init_r)
    x = Activation('relu')(x)
    x, l, c = googlenet_block(x)
    layers += l
    conv_n += c
    layers += 2
#=====================================================================================================
    if dropout[dropout_n] > 0.0:
        x = Dropout(dropout[dropout_n])(x)
        layers += 1
#=====================================================================================================
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x, l, c = googlenet_block(x)
    layers += l
    conv_n += c
    layers += 2
#=====================================================================================================
    if dropout[dropout_n] > 0.0:
        x = Dropout(dropout[dropout_n])(x)
        layers += 1
#=====================================================================================================    
    m = Add()([init_r, x])
    layers += 1
    return [m, layers, conv_n]
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
#===================================================================================================
def marge_resnet_googlenet():
    ip = Input(shape=init)
    layers = 1
    conv_n = 0
#=====================================================================================================
    if if_n[0]:
        for i in range(0, loop):
            x, l, c = resnet_block(ip, dropout_n = 0)
            layers += l
            conv_n += c
    x = LeakyReLU(alpha=config.alpha_lrelu)(x)
    x = Dropout(rate = dropout[0])(x)
    layers += 2
#=====================================================================================================
    if if_n[1]:
        for i in range(0, loop):
            x, l, c = resnet_block(ip, dropout_n = 0)
            layers += l
            conv_n += c
    x = LeakyReLU(alpha=config.alpha_lrelu)(x)
    x = Dropout(rate = dropout[0])(x)
    layers += 2
#=====================================================================================================
    if if_n[2]:
        for i in range(0, loop):
            x, l, c = resnet_block(ip, dropout_n = 0)
            layers += l
            conv_n += c
    x = LeakyReLU(alpha=config.alpha_lrelu)(x)
    x = Dropout(rate = dropout[0])(x)
    layers += 2
#=====================================================================================================
    x = AveragePooling2D((pool, pool))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)
    layers += 3

    model = Model(ip, x)
    return [model, layers, conv_n]

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
    model, layers, conv_n= marge_resnet_googlenet()
    model.summary()
    print("conv_n block  :", conv_n)
    print("Total layers  :", layers)
    print("loop , pool   :", loop,',', pool)
    print("conv2D shape  :", conv_2D)
    print("Dropout shape :", dropout)
    if gnet:
        print("GNet          : (1, 1)-->(n, n)")
    else:
        print("GNet          : (n, n)-->(1, 1)")

    return model

if __name__ == "__main__":
    model, layers, conv_n= marge_resnet_googlenet()
    model.summary()
    print("conv_n block  :", conv_n)
    print("Total layers  :", layers)
    print("loop , pool   :", loop,',', pool)
    print("conv2D shape  :", conv_2D)
    print("Dropout shape :", dropout)
    if gnet:
        print("GNet          : (1, 1)-->(n, n)")
    else:
        print("GNet          : (n, n)-->(1, 1)")

print("\n\n=========================================")
print("Model generate successfully.")
print("=========================================\n\n")

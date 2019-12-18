"""
Generating model.
This is the baseline model with the activation function controller
"""

import keras, os
from keras.layers import (Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU)
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from .. import config

model_checkpoint_dir = os.path.join(config.checkpoint_path(), "model.h5")
saved_model_dir = os.path.join(config.output_path(), config.test_model_name + ".h5")
#=========================================================================================
Conv_2D = [32, 32, 64, 64]
dropout = [0.20, 0.20, 0.30]
#=========================================================================================
def get_model():
    print("\n\n================ Activation function ================")
    print("input_ layer :", config.input_layer_activation)
    if config.alpha_lrelu:
        print("hidden layer : LeakyReLU")
    else:
        print("hidden layer :", config.hidden_layer_activation)
    print("output layer :", config.output_layer_activation)
    print("================ Activation function ================\n\n")

    model = Sequential()
    model.add(Conv2D(Conv_2D[0], (3, 3), padding = "same", input_shape = config.img_shape)) # 32
    # 2 hidden =============================================================================
    if config.alpha_lrelu:
        model.add(LeakyReLU(alpha=config.alpha_lrelu))
    else:
        model.add(Activation(config.hidden_layer_activation))
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    model.add(Conv2D(Conv_2D[1], (3, 3), padding = "same", input_shape = config.img_shape)) # 32
    # 3 hidden =============================================================================
    if config.alpha_lrelu:
        model.add(LeakyReLU(alpha=config.alpha_lrelu))
    else:
        model.add(Activation(config.hidden_layer_activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate = dropout[0])) #0.20
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    model.add(Conv2D(Conv_2D[2], (3, 3), padding = "same", input_shape = config.img_shape)) # 64
    # 4 hidden =============================================================================
    if config.alpha_lrelu:
        model.add(LeakyReLU(alpha=config.alpha_lrelu))
    else:
        model.add(Activation(config.hidden_layer_activation))
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    model.add(Conv2D(Conv_2D[3], (3, 3), padding = "same", input_shape = config.img_shape)) # 64
    # 5 hidden =============================================================================
    if config.alpha_lrelu:
        model.add(LeakyReLU(alpha=config.alpha_lrelu))
    else:
        model.add(Activation(config.hidden_layer_activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate = dropout[1])) # 0.20
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    model.add(Flatten())
    model.add(Dense(384, kernel_regularizer= keras.regularizers.l2(config.initial_weight)))
    # 6 hidden =============================================================================
    if config.alpha_lrelu:
        model.add(LeakyReLU(alpha=config.alpha_lrelu))
    else:
        model.add(Activation(config.hidden_layer_activation))
    model.add(Dropout(rate = dropout[2])) # 0.30
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    model.add(Dense(config.nb_classes))
    # 9 output =============================================================================
    model.add(Activation(config.output_layer_activation))

    model.summary()
    return model

#read mode
def read_model():
    model = load_model(saved_model_dir)
    return model

#save check-point model
def saved_model_checkpoint():
    return ModelCheckpoint(model_checkpoint_dir, 
                            monitor='val_loss', 
                            verbose=2, 
                            save_best_only=True, 
                            save_weights_only=False, 
                            mode='auto', 
                            period=1)

#early stopping of model
def set_early_stopping():
    return EarlyStopping(monitor='val_loss',
                            patience= 15,
                            verbose=2,
                            mode='auto')


if __name__ == "__main__":
    m = get_model()

print("\n\n=========================================")
print("Model generate successfully.")
print("=========================================\n\n")

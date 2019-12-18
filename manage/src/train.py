"""
Run model on train data
Epoch - action
"""
import keras
import numpy as np

from .. import config
from . import my_model, preprocess

x_train, y_train = preprocess.load_train_data()

print("train data shape : ", x_train.shape)
print("train data label : ", y_train.shape)

model = my_model.get_model()

model.compile(optimizer= my_model.model_optimizer,
            loss= keras.losses.categorical_crossentropy,
            metrics= ['accuracy'])

model_cp = my_model.saved_model_checkpoint()
lr_controller = my_model.reduce_lr()
early_stopping = my_model.set_early_stopping()

#-----------------------------outer scop------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
#data augmentation
datagen = ImageDataGenerator(
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                horizontal_flip=True,
                validation_split=config.validation_split,
        )
datagen.fit(x_train)

train_set = datagen.flow(x=x_train, y=y_train, batch_size=config.batch_size, subset='training', shuffle=True)
valid_set = datagen.flow(x=x_train, y=y_train, batch_size=config.batch_size, subset='validation', shuffle=True)

if config.earlyStopping_patience:
        model.fit_generator(train_set,
                epochs=config.nb_epochs,
                verbose=2,
                validation_data = valid_set,
                callbacks=[early_stopping, model_cp, lr_controller]
        )
else:
        model.fit_generator(train_set,
                epochs=config.nb_epochs,
                verbose=2,
                validation_data = valid_set,
                callbacks=[model_cp, lr_controller]
        )

"""
model.fit_generator(datagen.flow(x_train, y_train, batch_size=config.batch_size),
        steps_per_epoch=x_train.shape[0] // config.batch_size,
        epochs=config.nb_epochs,
        verbose=2,
        callbacks=[early_stopping, model_cp]
        )
"""
#-----------------------------outer scop------------------------------------------------
"""
model.fit(x_train, y_train,
         batch_size=config.batch_size,
         epochs=config.nb_epochs,
         verbose=2,
         shuffle= True,
         callbacks=[early_stopping, model_cp],
         validation_split=config.validation_split
         )
"""
print("\n\n=========================================")
print("Train finish successfully.")
print("=========================================\n\n")

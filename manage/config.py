"""
Create & Set paths.
Handle Parameters.
"""
import os
import keras

batch_size = 200
nb_epochs = 300

#model handling ==========================================================================
test_model_name = "model"
initial_weight = 0.01

alpha_lrelu = 0
if 0:
	alpha_lrelu = 0.1
input_layer_activation = "linear"
hidden_layer_activation = "relu"
output_layer_activation = "softmax"

# for baseline
Conv2D = [32, 64, 128]
Dropout = [0.10, 0.15, 0.20]

""" 
# for cnn 90
Conv2D = [32,32,  64,64,  128,128]
Dropout = [0.15, 0.20, 0.25, 0.30]
"""
monitor = 'val_loss'

#train handling ==========================================================================
earlyStopping_patience = 0
validation_split=0.25 # spliting for validation set

#preprocess handling =====================================================================
special_normalization = 1
#input settings ==========================================================================
nb_train_samples = 50000
nb_classes = 10
img_size = 32
img_channel = 3
img_shape = (img_size, img_size, img_channel)

nb_test_samples = 300000
nb_test_part = 6

#path functions ==========================================================================
def root_path():
	return os.path.dirname(__file__)

def checkpoint_path():
	return os.path.join(root_path(), "checkpoint")

def dataset_path():
	return os.path.join(root_path(), "dataset")

def output_path():
	return os.path.join(root_path(), "output")

def src_path():
	return os.path.join(root_path(), "src")

def submission_path():
	return os.path.join(root_path(), "submission")


print("\n\n=========================================")
print("Config finish successfully.")
print("=========================================\n\n")

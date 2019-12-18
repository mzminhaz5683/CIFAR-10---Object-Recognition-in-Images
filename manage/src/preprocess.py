"""
Load data.
Splite test data into parts.
"""

import numpy as np   
import cv2, os
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer

from .. import config

x_train = np.ndarray((config.nb_train_samples,
                    config.img_size, config.img_size,
                    config.img_channel), dtype=np.float32)

def normalization(x):
    # scale (0, 1)
    x = np.divide(x, 255.0)
    #mean = (x - .5)*2
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)

    return x 

# load train
def load_train_data():
    train_data_dir = os.path.join(config.dataset_path(), "train")
    train_images = sorted(os.listdir(train_data_dir), key=lambda x: int(x.split(".")[0]))
    train_images = [os.path.join(train_data_dir, img_path) for img_path in train_images]
    #print(train_images)

    train_labels_df = pd.read_csv(os.path.join(config.dataset_path(), "trainLabels.csv"))
    train_labels = train_labels_df["label"].values
    #print(train_labels, len(train_labels))

    encoder = LabelBinarizer()
    y_labels = encoder.fit_transform(train_labels)
    #print(train_labels[50])

    print("Loading train images ...")
    for i, img_dir in enumerate(train_images):
        img = cv2.imread(img_dir)
        img = normalization(img)
        #print(img)
        x_train[i] = img


    # calculating & storing : mean & std
    data_store = open(os.path.join(config.dataset_path(), "data_store.txt"), 'w')
    m = np.mean(x_train,axis=(0,1,2,3))
    s = np.std(x_train,axis=(0,1,2,3))
    mean = float("%.4f"%float(m))
    std= float("%.4f"%float(s))
    data_store.write(str(mean)+" "+str(std))
    print("mean : std =", mean, std)
    data_store.close()

    X_train = (x_train - mean)/(std + 1e-7)
    return X_train, y_labels


# load test data
def load_test_data():
    test_data_dir = os.path.join(config.dataset_path(), "test")
    test_images = sorted(os.listdir(test_data_dir), key=lambda x: int(x.split(".")[0]))
    test_images = [os.path.join(test_data_dir, img_path) for img_path in test_images]
    #print(test_images)

    #loading 
    nb_images = config.nb_test_samples//config.nb_test_part
    #print("per part : ", nb_images)
    start = 0
    end = nb_images
    for part in range(0, config.nb_test_part):
        if not (part == 0): start += nb_images
        end = start + nb_images
        #print(start, ' : ', end)

        x_test = np.ndarray((nb_images,
                        config.img_size, config.img_size,
                        config.img_channel), dtype=np.float32)

        print("Loading test images from ", start, " to ", end)
        for i, img_dir in enumerate(test_images[start:end]):
            img = cv2.imread(img_dir)
            img = normalization(img)
            #print(img)
            x_test[i] = img

        data_store = open(os.path.join(config.dataset_path(), "data_store.txt"), 'r')
        ls = data_store.readlines()
        a, b = ls[0].split(" ")
        mean = float(a)
        std = float(b)
        data_store.close()
        print("mean : std =", mean, std)

        X_test = (x_test - mean)/(std + 1e-7)
        np.save(os.path.join(config.output_path(), "test_parts", "x_test_"+ str(part)), X_test)
        del x_test


def get_test_data_by_part(part):
    print("loading test image part", str(part), "from numpy array")
    return np.load(os.path.join(config.output_path(), "test_parts", "x_test_"+ str(part) + ".npy"))

if __name__ == "__main__": 
    x, y = load_train_data()
    print("Train shape : ", x.shape)
    print("Train level : ", y.shape)

    load_test_data()
    get_test_data_by_part(0)
    
    print("\n\n=========================================")
    print("Preprocess finish successfully.")
    print("=========================================\n\n")
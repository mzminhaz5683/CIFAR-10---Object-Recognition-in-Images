import numpy as np 
import pandas as pd 
import os
from keras.models import load_model
from statistics import mode
from sklearn.ensemble import VotingClassifier

from .. import config
from . import preprocess, my_model

# emsamble
models = ["8711", "8567", "8562"]
default = 3
#=================================================================================================
label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
lst_results = []
for i in models:
    print("\nWork on model : ", i+ ".h5")
    saved_model_dir = os.path.join(config.output_path(), "marge_model", i + ".h5")
    model = load_model(saved_model_dir)
    #=================================================================================================
    result = []
    for part in range(0, config.nb_test_part):
        x_test = preprocess.get_test_data_by_part(part)

        print("Predicting results")
        predictions = model.predict(x_test, batch_size = config.batch_size, verbose = 2)

        #print(len(predictions))
        #print(predictions.shape)

        label_pred = np.argmax(predictions, axis = 1)
        #print(label_pred)

        #convet numpy vector into list
        result += label_pred.tolist()
    #=================================================================================================
    lst_results.append(result)
#====================================================================================================
#====================================================================================================
print("\n\nMode activate. please wait.....")
result = np.array([])
for i in range(0, config.nb_test_samples):
    try:
        result = np.append(result, mode([ lst[i] for lst in lst_results  ]))
    except:
        result = np.append(result, lst_results[default][i])
"""     
    if i == 10:
        break
result = [label[int(i)] for i in result]
print(result)
"""
#====================================================================================================
if 1:
    result = [label[int(i)] for i in result]
    submit_df = pd.DataFrame({"id": range(1, config.nb_test_samples+1),
                    "label": result})
                
    submit_df.to_csv(os.path.join(config.submission_path(), "submission_mz.csv"),
                    header=True, index=False)

print("\n\n=========================================")
print("Submission file saved Sucessfully.")
print("=========================================\n\n")
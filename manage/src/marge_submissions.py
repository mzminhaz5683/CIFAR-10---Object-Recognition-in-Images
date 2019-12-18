import numpy as np 
import pandas as pd 
import os
from statistics import mode

from .. import config
lst_sub = ["submission_mz", "87390","88910"]
default = 0
# "86160", "86970"
#===================================================================================================
label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
dummy = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

load_sub = []
for i in lst_sub:
    try:
        if i == "submission_mz":
            load_sub.append(pd.read_csv(os.path.join(config.submission_path(), i+".csv")))
        else:
            load_sub.append(pd.read_csv(os.path.join(config.submission_path(), "marge_sbumission", i+ ".csv")))
    except Exception as e:
        print("Excaption called for :: ", e)
        continue

load_label = []
for lst in load_sub:
    load_label.append([lst.iloc[:,1]])

d_load = []
for i in load_label:   
    d_load.append(i[0].map(dummy))
    
print("\n\nMode activate. please wait.....")
result = np.array([])

for i in range(0, config.nb_test_samples):
    try:
        result = np.append(result, mode([ lst[i] for lst in d_load  ]))
    except:
        result = np.append(result, d_load[default][i])
"""        
    if i == 10:
        break
result = [label[int(i)] for i in result]
print(result)
"""
if 1:
    result = [label[int(i)] for i in result]
    submit_df = pd.DataFrame({"id": range(1, config.nb_test_samples+1),
                    "label": result})
                
    submit_df.to_csv(os.path.join(config.submission_path(), "submission_mz.csv"),
                    header=True, index=False)

print("\n\n=========================================")
print("Submission file saved Sucessfully.")
print("=========================================\n\n")
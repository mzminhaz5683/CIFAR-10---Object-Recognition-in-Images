import numpy as np 
import pandas as pd 
import os

from .. import config
from . import preprocess, my_model

model = my_model.read_model()

label = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]


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


#sumbit into csv
result = [label[i] for i in result]
#print(result)

submit_df = pd.DataFrame({"id": range(1, config.nb_test_samples+1),
                "label": result})
            
submit_df.to_csv(os.path.join(config.submission_path(), "submission_mz.csv"),
                header=True, index=False)

print("\n\n=========================================")
print("Submission file saved Sucessfully.")
print("=========================================\n\n")
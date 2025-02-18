### Script to run classification pipeline on test data ###
## loads models from saved file and runs textclassification pipeline
## the predicetd labels are then saved to a file. 
## the file is then used to evaluate the performance of the model

import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pickle as pkl
import pandas as pd

# load data
path = "dataset.txt" # text file
with open(path, "rb") as f:
    dictionary = pkl.load(f)

data = dictionary["raw_text"]

# load models
output_dir = "datapath"

model = AutoModelForSequenceClassification.from_pretrained(output_dir) 
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# run pipeline
pipe = TextClassificationPipeline(model =model, tokenizer=tokenizer, return_all_scores=True)
predicted_labs = pipe(data)

predicted_dict = {}
themes = ["Comorbidities", "Physical Health", "Pyschological & emotional", "Daily Life", "Social Support", "Health Pathways & services"]

for x in range(len(themes)):
    predicted_dict[themes[x]] = [predicted_labs[i][x]['score'] for i in range(0, len(predicted_labs))]

pred_df = pd.DataFrame(predicted_dict)

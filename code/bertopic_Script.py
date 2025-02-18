from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score, 
                             adjusted_mutual_info_score, adjusted_rand_score,
                               homogeneity_score, completeness_score,
                                 homogeneity_completeness_v_measure, accuracy_score)
from bertopic import vectorizers

import pandas as pd
import numpy as np
import pickle as pkl
import regex as re


def select_data(dataf, test_data, col_data ,col_test):
    # dataf[col_data].apply(lambda x: x.lower())
    labelled = dataf.loc[dataf[col_data].apply(lambda x: x.lower()).isin(test_data[col_test].apply(lambda x: x.lower()))]
    labelled = labelled.drop_duplicates(subset=col_data)
    labelled = labelled.sort_values(by=[col_data])
    
    return labelled


# load seed terms

dump_dir = "datapath"
seeds_cor = pd.read_excel(dump_dir + "seed_terms.xlsx", sheet_name= "seed_terms")
seed_cor = seeds_cor.iloc[1:,1:]
seed_cor.columns = seeds_cor.iloc[0,1:]
seeds_cor = [row.lower().split(", ") for row in seed_cor.iloc[:,1]]

seeds_corpus = []
for row in seeds_cor:
    seeds = [re.sub(r'[^\w\s]','', elem) for elem in row]
    seeds_corpus.append(seeds)

# load data
data = pd.read_excel(dump_dir + "stopwords_anom_removed.xlsx")

comment = data.comments_raw
data = data.dropna(subset=["stopwords_removed"])
comments = data.comments
docs = data.stopwords_removed.astype(str)
docs

# run bertopic
classidf = vectorizers.ClassTfidfTransformer(reduce_frequent_words=True)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(list(docs), show_progress_bar=False)

# Run bertopic for each set of seed terms 
predictions = {}
themes = ["Comorbidities", "Physical Health", "Pyschological & emotional", "Daily Life", "Social Support", "Cancer Pathways & services"]


for i, seeds in enumerate(seeds_corpus):
    topic_model=  BERTopic(seed_topic_list=seeds, nr_topics=1, n_gram_range=(1,2), verbose=True, calculate_probabilities=False, ctfidf_model=classidf)
    topic_model= topic_model.fit(list(docs))
    # Save results
    pred_test = np.asarray(topic_model)
    predictions[themes[i]] = list(map(float, pred_test))

predictions_df = pd.DataFrame.from_dict(predictions, orient='columns')
predictions_df.head()
predictions_df.to_csv("prediction_bertopic.csv")


predictions = {}
themes = ["Comorbidities", "Physical Health", "Pyschological & emotional", "Daily Life", "Social Support", "Cancer Pathways & services"]
pkl_dir = r"/bertopic/out/"

for i in range(6):
    print(themes[i])
    with open(pkl_dir + "bertopic_pca_out_{}.pkl".format(i), "rb") as fp:
        topic_prob = pkl.load(fp)
        predictions[themes[i]] = list(map(float, topic_prob))

predictions_df = pd.DataFrame.from_dict(predictions, orient='columns')
predictions_df.head()


print("saving df")

predictions_df.to_csv("prediction_bertopic.csv")


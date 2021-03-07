import pandas as pd
import numpy as np
from pprint import pprint
import csv
from dicttoxml import dicttoxml
import sys
import infoGain
file_name = sys.argv[1]
output_file = sys.argv[2]


with open(file_name) as csv_file:
    data_list = list(csv.reader(csv_file))

m = len(data_list)
n = len(data_list[1])

att = []

for attr in range(0,n):
    att.append('att'+ str(attr))


dataset = pd.read_csv(file_name,names=att)



def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):


    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]


    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]


    elif len(features) == 0:
        return parent_node_class



    else:

        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]


        item_values = [infoGain.InfoGain(data, feature, att[-1]) for feature in features]
        entropy_values = []
        entropy_values.append([infoGain.entropyList(data, feature, att[-1]) for feature in features])
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]


        tree = {best_feature:{}}


        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value

            sub_data = data.where(data[best_feature] == value).dropna()


            subtree = ID3(sub_data, dataset, features, att[-1], parent_node_class)


            tree[best_feature][value] = subtree


        return (tree)


tree = ID3(dataset,dataset,dataset.columns[:-1],att[-1])

xml = dicttoxml(tree)

with open(output_file, "wb") as f:
    f.write(xml)


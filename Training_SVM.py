import os
import shutil

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import math
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib.widgets import Slider
import winsound
import xlsxwriter
from openpyxl import load_workbook
import os.path
import csv
from scipy.stats import shapiro
import json
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):

    # Show more than 10 or 20 rows when a dataframe comes back.
    pd.set_option('display.max_rows', max_rows)
    # Columns displayed in debug view
    pd.set_option('display.max_columns', max_columns)
    pd.options.display.float_format = '{:.2f}'.format
    pd.set_option('display.width', display_width)


# run
pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320)


def IsInAnyCluster(clusterRange,num):
    labels_length = []
    for range in clusterRange:
        if num <= range['normalized_max'] and num >= range['normalized_min']:
            inner_list={}
            inner_list['label'] = range['label'] + 1
            inner_list['length'] = range['normalized_max'] - range['normalized_min']
            labels_length.append(inner_list)
    if len(labels_length)> 0:
        max_item = min(labels_length, key=lambda x: x['length'])
        return max_item['label']
    return 0



################################################################################################ Constants

clusterRange = [{'label': 0, 'countitems_bforekde': 8867, 'min': 62100733560, 'max': 162891145646, 'countitems_afterkde': 8494, 'normalized_min': 65833711785, 'normalized_max': 160873319578, 'isgaussian': 0, 'countms': 95002023918}, {'label': 1, 'countitems_bforekde': 31, 'min': 67338621770, 'max': 134389442895, 'countitems_afterkde': 22, 'normalized_min': 78278845837, 'normalized_max': 119556378361, 'isgaussian': 1, 'countms': 35508333193}, {'label': 2, 'countitems_bforekde': 27, 'min': 79987144792, 'max': 142633128416, 'countitems_afterkde': 24, 'normalized_min': 88703653023, 'normalized_max': 138243519953, 'isgaussian': 1, 'countms': 42739514093}, {'label': 3, 'countitems_bforekde': 21, 'min': 99535054916, 'max': 141292743949, 'countitems_afterkde': 21, 'normalized_min': 99535054916, 'normalized_max': 141292743949, 'isgaussian': 1, 'countms': 41757689033}]



benign_capture_date_start = 1665100800000000
benign_capture_date_end = 1665187200000000

attack_capture_date_start = 1666656000000000
attack_capture_date_end = 1666742400000000

LabelColumn = 'label'
feature_columns =[]
StartTime = datetime.today()
IsFeatureApplied = 0
output_file='C:\\CICIoT2023\\train\\log.csv'

################################################################################################ Load Dataframe

filtered_df = pd.read_csv("C:\\CICIoT2023\\csv\\mix\\all_records_20k_ACK_Fragmentation.csv")

################################################################################################ Add New Label to DataFrame

filtered_df.insert(filtered_df.columns.get_loc(LabelColumn), "TimeCluster",0)
filtered_df['TimeCluster'] = filtered_df['Timestamp'].apply(lambda x:IsInAnyCluster(clusterRange,x))
IsFeatureApplied = 1
# print(filtered_df['TimeCluster'].value_counts())
# exit()
################################################################################################ Dataframe preprocessing

feature_columns = filtered_df.columns.tolist()
# feature_columns.remove(LabelColumn)
# feature_columns.remove('HTTP')
# feature_columns.remove('HTTPS')
# feature_columns.remove('TCP')
# feature_columns.remove('Protocol Type')
# feature_columns.remove('Protocol_name')
# feature_columns.remove('UDP')




labelencoder_X = LabelEncoder()
filtered_df['Protocol_name'] = labelencoder_X.fit_transform(filtered_df['Protocol_name'])
X = filtered_df.loc[:, feature_columns].values
y = filtered_df.loc[:, LabelColumn].values
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = SimpleImputer.fit(imputer,X[:,:])
X[:, :] = imputer.transform(X[:, :])
X[:, :] = imputer.transform(X[:, :])


################################################################################################ Feature selection

from sklearn.feature_selection import SelectKBest, f_classif

#X_selected = X
selector = SelectKBest(f_classif, k=10)
selector.fit_transform(X, y)
# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_features = list(selected_feature_indices)

# selected_features.remove(0)
# selected_features.remove(25)
# selected_features.remove(26)
# selected_features.remove(26)
# selected_features.remove(57)

feature_importance_dict = dict(zip(feature_columns, selector.scores_))
print(selected_features)

res_list = [feature_columns[i] for i in selected_features]
print(res_list)
# print(selector.scores_)
print(feature_importance_dict)
X_selected = X[:, selected_features]
print(X_selected[1,:])
################################################################################################ Dataframe splitting

X_train, X_test, y_train, y_test = train_test_split(X_selected,y, test_size=0.20)

################################################################################################ Training


#clf = pickle.load(open('Models\\AllAttacks_MIX_56k_FeatureImportanceBalancedAfterSMOTE_Test_Balanced.sav', 'rb'))
#classifier_predictions = clf.predict(X_test)

print(str(datetime.today())+": Start training")
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)
print(str(datetime.today())+": Training done")
print(str(datetime.today())+": Start testing")
classifier_predictions = model.predict(X_test)
print(str(datetime.today())+": Start testing done")

precision_metric = precision_score(y_test, classifier_predictions, average = "macro")
print("precision_metric: " + str(precision_metric))


recall_metric = recall_score(y_test, classifier_predictions, average = "macro")
print("recall_metric: " + str(recall_metric))

accuracy_metric = accuracy_score(y_test, classifier_predictions)
print("accuracy_metric: " + str(accuracy_metric))

f1_metric = f1_score(y_test, classifier_predictions, average = "macro")
print("f1_metric: " + str(f1_metric))


scores = cross_val_score(model, X_selected, y, cv=5)
print("scores are")
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


EndTime = datetime.today()

with open(output_file, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    data_row = [StartTime.strftime("%Y-%m-%d_%H:%M:%S"),EndTime.strftime("%Y-%m-%d_%H:%M:%S"),"SVM",IsFeatureApplied,precision_metric,recall_metric,accuracy_metric,f1_metric,scores.mean(),scores.std()]
    csv_writer.writerow(data_row)

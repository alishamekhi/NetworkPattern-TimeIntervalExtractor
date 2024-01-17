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


def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):

    # Show more than 10 or 20 rows when a dataframe comes back.
    pd.set_option('display.max_rows', max_rows)
    # Columns displayed in debug view
    pd.set_option('display.max_columns', max_columns)

    pd.set_option('display.width', display_width)


# run
pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320)



benign_capture_date_start = 1665100800000000
benign_capture_date_end = 1665187200000000

attack_capture_date_start = 1663113600000000
attack_capture_date_end = 1663200000000000


############################################################################### Benign
df_benign = pd.read_csv("C:\\CICIoT2023\\csv\\benign\\BenignTraffic_Merged_01percent.csv")
column_name = 'Timestamp'  # Replace with your actual column name
df_benign.columns = df_benign.columns.str.strip()
df_benign = df_benign.replace([np.inf,-np.inf], np.nan)


if 'ts' in df_benign.columns:
    df_benign.insert(df_benign.columns.get_loc("ts") + 1, column_name,(df_benign['ts']*1000000)-benign_capture_date_start , True)
    df_benign[column_name] = df_benign[column_name].astype('Int64')
    df_benign = df_benign.drop(columns=['ts'])
if 'IAT' in df_benign.columns:
    df_benign = df_benign.drop(columns=['IAT'])
if 'max_duration' in df_benign.columns:
    df_benign = df_benign.drop(columns=['max_duration'])
if 'min_duration' in df_benign.columns:
    df_benign = df_benign.drop(columns=['min_duration'])
if 'average_duration' in df_benign.columns:
    df_benign = df_benign.drop(columns=['average_duration'])

benign_min_value = df_benign[column_name].min()
benign_max_value = df_benign[column_name].max()

benign_timedist = (benign_max_value-benign_min_value)/8

filtered_df_benign = df_benign
# filtered_df = df_benign[
#     ((df_benign[column_name] > benign_min_value+0*benign_timedist) & (df_benign[column_name] <= benign_min_value+1*benign_timedist))|
#     ((df_benign[column_name] > benign_min_value+2*benign_timedist) & (df_benign[column_name] <= benign_min_value+3*benign_timedist))|
#     ((df_benign[column_name] > benign_min_value+4*benign_timedist) & (df_benign[column_name] <= benign_min_value+5*benign_timedist))|
#     ((df_benign[column_name] > benign_min_value+6*benign_timedist) & (df_benign[column_name] <= benign_min_value+7*benign_timedist))
#       ]

filtered_df_benign['label'] = 0

X_benign = filtered_df_benign.loc[:, :].values
benign_labelencoder_X = LabelEncoder()
X_benign[:, filtered_df_benign.columns.get_loc("Protocol_name")] = benign_labelencoder_X.fit_transform(X_benign[:, filtered_df_benign.columns.get_loc("Protocol_name")])
benign_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
benign_imputer = SimpleImputer.fit(benign_imputer,X_benign[:,:])
X_benign[:, :] = benign_imputer.transform(X_benign[:, :])
X_benign[:, :] = benign_imputer.transform(X_benign[:, :])


############################################################################### Attack

df_attack = pd.read_csv("C:\\CICIoT2023\\csv\\attack\\DDoS-HTTP_Flood_035percent.csv")
column_name = 'Timestamp'  # Replace with your actual column name
df_attack.columns = df_attack.columns.str.strip()
df_attack = df_attack.replace([np.inf,-np.inf], np.nan)


if 'ts' in df_attack.columns:
    df_attack.insert(df_attack.columns.get_loc("ts") + 1, column_name,(df_attack['ts']*1000000)-attack_capture_date_start , True)
    df_attack[column_name] = df_attack[column_name].astype('Int64')
    df_attack = df_attack.drop(columns=['ts'])
if 'IAT' in df_attack.columns:
    df_attack = df_attack.drop(columns=['IAT'])
if 'max_duration' in df_attack.columns:
    df_attack = df_attack.drop(columns=['max_duration'])
if 'min_duration' in df_attack.columns:
    df_attack = df_attack.drop(columns=['min_duration'])
if 'average_duration' in df_attack.columns:
    df_attack = df_attack.drop(columns=['average_duration'])

attack_min_value = df_attack[column_name].min()
attack_max_value = df_attack[column_name].max()

attack_timedist = (attack_max_value-attack_min_value)/8

filtered_df_attack = df_attack
# filtered_df = df_attack[
#     ((df_attack[column_name] > attack_min_value+0*attack_timedist) & (df_attack[column_name] <= attack_min_value+1*attack_timedist))|
#     ((df_attack[column_name] > attack_min_value+2*attack_timedist) & (df_attack[column_name] <= attack_min_value+3*attack_timedist))|
#     ((df_attack[column_name] > attack_min_value+4*attack_timedist) & (df_attack[column_name] <= attack_min_value+5*attack_timedist))|
#     ((df_attack[column_name] > attack_min_value+6*attack_timedist) & (df_attack[column_name] <= attack_min_value+7*attack_timedist))
#       ]

filtered_df_attack['label'] = 1
X_attack = filtered_df_attack.loc[:, :].values
attack_labelencoder_X = LabelEncoder()
X_attack[:, filtered_df_attack.columns.get_loc("Protocol_name")] = attack_labelencoder_X.fit_transform(X_attack[:, filtered_df_attack.columns.get_loc("Protocol_name")])
attack_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
attack_imputer = SimpleImputer.fit(attack_imputer,X_attack[:,:])
X_attack[:, :] = attack_imputer.transform(X_attack[:, :])
X_attack[:, :] = attack_imputer.transform(X_attack[:, :])


##########################################################################################

output = "C:\\CICIoT2023\\csv\\mix\\all_records_20k_DDoS-HTTP_Flood.csv"
print('Benign Records: ' + str(len(filtered_df_benign)))
print('Attack Records: ' + str(len(filtered_df_attack)))

df_all_records = pd.concat([filtered_df_benign,filtered_df_attack],ignore_index=True)
df_all_records.sort_values(by=column_name, inplace=True)

df_all_records.to_csv(output,index=False)

print("Saved in: " + output)
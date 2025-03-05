import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
from tsfresh import extract_features

from utils.helpers import *
from utils.params import *

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

setSeed()



battery_columns = [
    'soc',
    'soh',
    'regenwh',
    'Motor Pwr(w)',
    'Aux Pwr(100w)',
    'Motor Temp',
    'Torque Nm',
    'rpm',
    'capacity',
    'ref_consumption'
]

dataset_path = './dataset/real/'
dacia_path = './dataset/real/DACIA SPRING/'
nissan_path = './dataset/real/NISSAN LEAF/'

dfs = []

for file in os.listdir(dacia_path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(dacia_path, file))
        dfs.append(df)

for file in os.listdir(nissan_path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(nissan_path, file))
        dfs.append(df)

### DRIVER IDENTIFICATION ###
tsfreshs = []
window_size = 10

save_path = os.path.join(dataset_path, 'driver.parquet')

if not os.path.exists(save_path):
    for df in tqdm(dfs, desc='Extracting Features', total=len(dfs)):
        driver = df['driver']
        df = df[battery_columns]
        df['driver'] = driver
        df.dropna(inplace=True)

        for start_idx in range(0, len(df), window_size):
            end_idx = min(start_idx + window_size, len(df))
            window_df = df.iloc[start_idx:end_idx]

            if not window_df.empty:
                extracted_features = extract_features(window_df, column_id='driver', n_jobs=os.cpu_count(), disable_progressbar=True)
                extracted_features['driver'] = driver.iloc[0]
                tsfreshs.append(extracted_features)
    
    tsfresh = pd.concat(tsfreshs)
    tsfresh.to_parquet(save_path)
else:
    tsfresh = pd.read_parquet(save_path)

driver = tsfresh['driver']
tsfresh = tsfresh.drop(columns=['driver'])
tsfresh = impute(tsfresh)
tsfresh = select_features(tsfresh, driver)

X_train, X_test, y_train, y_test = train_test_split(tsfresh, driver, test_size=0.2, random_state=seed)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

results = []

for name, clf, param in tqdm(zip(names, classifiers, parameters), desc='Driver Identification', total=len(names)):
    # GridSearchCV
    accuracy_driver, f1_driver = trainGrid(clf, param, X_train, y_train, X_test, y_test)
    results.append([name, accuracy_driver, f1_driver])

results_df = pd.DataFrame(results, columns=['model', 'accuracy_driver', 'f1_driver'])
results_df.to_csv('./results/real/driver_identification.csv', index=False)


### STARTING POINT IDENTIFICATION ###

tsfreshs = []
window_size = 5

save_path = os.path.join(dataset_path, 'start.parquet')

if not os.path.exists(save_path):
    for df in tqdm(dfs, desc='Extracting Features', total=len(dfs)):
        # Taking only first 25% of the data
        df = df.iloc[:int(len(df) * 0.25)]
        start = df['route_code'].unique()[0].split('_')[2]
        df = df[battery_columns]
        df['start'] = start
        df.dropna(inplace=True)

        for start_idx in range(0, len(df), window_size):
            end_idx = min(start_idx + window_size, len(df))
            window_df = df.iloc[start_idx:end_idx]

            if not window_df.empty:
                extracted_features = extract_features(window_df, column_id='start', n_jobs=os.cpu_count(), disable_progressbar=True)
                extracted_features['start'] = start
                tsfreshs.append(extracted_features)
    
    tsfresh = pd.concat(tsfreshs)
    tsfresh.to_parquet(save_path)
else:
    tsfresh = pd.read_parquet(save_path)

start = tsfresh['start']
tsfresh = tsfresh.drop(columns=['start'])
tsfresh = impute(tsfresh)
tsfresh = select_features(tsfresh, start)

X_train, X_test, y_train, y_test = train_test_split(tsfresh, start, test_size=0.2, random_state=seed)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

results = []

for name, clf, param in tqdm(zip(names, classifiers, parameters), desc='Start Identification', total=len(names)):
    # GridSearchCV
    start_driver, start_driver = trainGrid(clf, param, X_train, y_train, X_test, y_test)
    print(f'{name} {start_driver} {start_driver}')
    results.append([name, start_driver, start_driver])

results_df = pd.DataFrame(results, columns=['model', 'start_driver', 'start_driver'])
results_df.to_csv('./results/real/start_identification.csv', index=False)


### ENDING POINT IDENTIFICATION ###

tsfreshs = []
window_size = 5

save_path = os.path.join(dataset_path, 'end.parquet')

if not os.path.exists(save_path):
    for df in tqdm(dfs, desc='Extracting Features', total=len(dfs)):
        # Taking only last 25% of the data
        df = df.iloc[int(len(df) * 0.75):]
        end = df['route_code'].unique()[0].split('_')[3]
        df = df[battery_columns]
        df['end'] = end
        df.dropna(inplace=True)

        for start_idx in range(0, len(df), window_size):
            end_idx = min(start_idx + window_size, len(df))
            window_df = df.iloc[start_idx:end_idx]

            if not window_df.empty:
                extracted_features = extract_features(window_df, column_id='end', n_jobs=os.cpu_count(), disable_progressbar=True)
                extracted_features['end'] = end
                tsfreshs.append(extracted_features)
    
    tsfresh = pd.concat(tsfreshs)
    tsfresh.to_parquet(save_path)
else:
    tsfresh = pd.read_parquet(save_path)

end = tsfresh['end']
tsfresh = tsfresh.drop(columns=['end'])
tsfresh = impute(tsfresh)
tsfresh = select_features(tsfresh, end)

X_train, X_test, y_train, y_test = train_test_split(tsfresh, end, test_size=0.2, random_state=seed)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

results = []

for name, clf, param in tqdm(zip(names, classifiers, parameters), desc='End Identification', total=len(names)):
    # GridSearchCV
    end_driver, end_driver = trainGrid(clf, param, X_train, y_train, X_test, y_test)
    print(f'{name} {end_driver} {end_driver}')
    results.append([name, end_driver, end_driver])

results_df = pd.DataFrame(results, columns=['model', 'end_driver', 'end_driver'])
results_df.to_csv('./results/real/end_identification.csv', index=False)
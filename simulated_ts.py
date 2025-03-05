import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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



dataset_path = './dataset/simulated/'
results_path = './results/simulated/'
os.makedirs(results_path, exist_ok=True)

files = []

# Loading all file paths
for subfolder in sorted(os.listdir(dataset_path)):
    subfolder_path = os.path.join(dataset_path, subfolder)
    subfolder_files = []
    if os.path.isdir(subfolder_path):
        for file in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, file)
            if file.endswith('.csv'):
                subfolder_files.append(file_path)
    if len(subfolder_files) > 0:
        files.append(subfolder_files)


for j, subfolder in enumerate(files):
    # Results list to store the model and accuracies
    results = []
    results_path = f'./results/simulated/{subfolder[0].split("/")[3]}.csv'
    if not os.path.exists(results_path):
        dfs = []
        parquet_path = '/'.join(subfolder[0].split('/')[:-1]) + '/tsfresh.parquet'
        if not os.path.exists(parquet_path):
                for file in tqdm(subfolder, desc=f'Loading {j+1}/{len(files)}', total=len(subfolder)):
                    try:
                        df = pd.read_csv(file, sep=';')
                        df.dropna(inplace=True)
                        df = df[batttery_simulated + ['vehID']]
                        df = df[df['vehID'].notna() & df['vehID'].str.startswith('EV')]
                        # Appending labels
                        file_name = file.split('/')[-1]
                        traffic = file_name.split('_')[3]
                        occupancy = file_name.split('_')[4]
                        auxiliaries = file_name.split('_')[5]
                        wind = file_name.split('_')[6]
                        # Tsfresh processing
                        df['vehID_group'] = df['vehID'].apply(lambda x: f'EV{int(x[2:]) // 3}')
                        grouped = df.groupby('vehID_group')
                        groups = []
                        for name, group in grouped:
                            group = group.iloc[:100]
                            group['group_id'] = name
                            # Extracting features using tsfresh
                            group = group.drop(columns=['vehID', 'vehID_group'])
                            extracted_features = extract_features(group, column_id='group_id', n_jobs=os.cpu_count(), disable_progressbar=True)
                            # Reattaching labels
                            extracted_features['traffic'] = traffic
                            extracted_features['occupancy'] = occupancy
                            extracted_features['auxiliaries'] = auxiliaries
                            extracted_features['wind'] = wind
                            groups.append(extracted_features)
                        dfs.append(pd.concat(groups))
                    except:
                        continue
        else:
            df = pd.read_parquet(parquet_path)
            dfs.append(df)
        # Concatenating all dataframes
        df = pd.concat(dfs)
        if not os.path.exists(parquet_path):
            # Save the concatenated dataframe as a parquet file
            df.to_parquet(parquet_path)
        # Getting labels
        traffic = df['traffic']
        occupancy = df['occupancy']
        auxiliaries = df['auxiliaries']
        wind = df['wind']
        # Impute missing values in the extracted features
        df = df.drop(columns=['traffic', 'occupancy', 'auxiliaries', 'wind'])
        impute(df)
        # Select relevant features based on label
        df_traffic = select_features(df, traffic)
        df_occupancy = select_features(df, occupancy)
        df_auxiliaries = select_features(df, auxiliaries)
        df_wind = select_features(df, wind)
        # Train test split
        X_train, X_test, y_traffic_train, y_traffic_test = train_test_split(df, traffic, test_size=0.2, random_state=seed)
        X_train, X_test, y_occupancy_train, y_occupancy_test = train_test_split(df, occupancy, test_size=0.2, random_state=seed)
        X_train, X_test, y_auxiliaries_train, y_auxiliaries_test = train_test_split(df, auxiliaries, test_size=0.2, random_state=seed)
        X_train, X_test, y_wind_train, y_wind_test = train_test_split(df, wind, test_size=0.2, random_state=seed)

        

        # Classification for driving style and car model
        for name, clf, param in tqdm(zip(names, classifiers, parameters), desc=f'Training {j+1}/{len(files)}', total=len(names)):
            # GridSearchCV
            accuracy_traffic, f1_traffic = trainGrid(clf, param, X_train, y_traffic_train, X_test, y_traffic_test)
            accuracy_occupancy, f1_occupancy = trainGrid(clf, param, X_train, y_occupancy_train, X_test, y_occupancy_test)
            accuracy_auxiliaries, f1_auxiliaries = trainGrid(clf, param, X_train, y_auxiliaries_train, X_test, y_auxiliaries_test)
            accuracy_wind, f1_wind = trainGrid(clf, param, X_train, y_wind_train, X_test, y_wind_test)
            results.append([name, accuracy_traffic, f1_traffic, accuracy_occupancy, f1_occupancy, accuracy_auxiliaries, f1_auxiliaries, accuracy_wind, f1_wind])


        results_df = pd.DataFrame(results, columns=['model',
                                                    'accuracy_traffic', 'f1_traffic',
                                                    'accuracy_occupancy', 'f1_occupancy',
                                                    'accuracy_auxiliaries', 'f1_auxiliaries',
                                                    'accuracy_wind', 'f1_wind'])
        results_df.to_csv(results_path, index=False)
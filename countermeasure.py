import os
import pandas as pd
import sys
from tqdm import tqdm
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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

K = 30
sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for j, subfolder in enumerate(files):
    # Stratified reduction
    sampled_files = subfolder[::K] 
    for i, file_path in enumerate(sampled_files):
        for size in tqdm(sizes, desc=f'{j+1}/{len(files)} | {i+1}/{len(sampled_files)}', total=len(sizes)):
            # Check if already processed
            results_path = file_path.replace('dataset', 'countermeasure')
            results_path = results_path.replace('/simulated/', f'/simulated/{size}/')
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if not os.path.exists(results_path):
                try:
                    # Load the dataset
                    df = pd.read_csv(file_path, sep=';')
                    # Keeping only nominal values
                    df = df[df['vehID'].notna() & df['vehID'].str.startswith('EV')]
                    # Keeping only battery-related features
                    vehID = df['vehID']
                    df = df[batttery_simulated]
                    # Adding driving style and vehicle label
                    car_model = vehID.map(lambda x: vehicle_info[x][0])
                    driving_style = vehID.map(lambda x: vehicle_info[x][1])
                    # Time-series aggregation on windows of size samples
                    df = df.groupby(df.index // size).mean()
                    vehID = vehID.groupby(vehID.index // size).first().reset_index(drop=True)
                    car_model = car_model.groupby(car_model.index // size).first().reset_index(drop=True)
                    driving_style = driving_style.groupby(driving_style.index // size).first().reset_index(drop=True)
                    df['car_model'] = car_model
                    df['driving_style'] = driving_style
                    # Defining train and test sets
                    X = df.drop(['car_model', 'driving_style'], axis=1)
                    y_style = df['driving_style']
                    y_car = df['car_model']
                    X_train, X_test, y_style_train, y_style_test = train_test_split(X, y_style, test_size=0.2, random_state=seed)
                    X_train, X_test, y_car_train, y_car_test = train_test_split(X, y_car, test_size=0.2, random_state=seed)
                    
                    # Results list to store the model and accuracies
                    results = []

                    # Classification for driving style and car model
                    for name, clf, param in zip(names, classifiers, parameters):#, desc=f'{j+1}/{len(files)} | {i+1}/{len(sampled_files)} | {k+1}/{len(sizes)}', total=len(names)):
                        # GridSearch
                        accuracy_style, f1_style = trainGrid(clf, param, X_train, y_style_train, X_test, y_style_test)
                        accuracy_car, f1_car = trainGrid(clf, param, X_train, y_car_train, X_test, y_car_test)
                        # Store results for the current model
                        results.append([name, accuracy_style, f1_style, accuracy_car, f1_car])
                    
                    # Save the results to a CSV file for the current dataset
                    results_df = pd.DataFrame(results, columns=['model', 'style_accuracy', 'style_f1', 'car_accuracy', 'car_f1'])
                    results_df.to_csv(results_path, index=False)
                except:
                    continue
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch



# Seed for reproducibility
seed = 151836

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mapping for vehicle IDs to car models and driving styles
vehicle_info = {
    'EV0': ('BMW_i3', 'defensive style'),
    'EV1': ('BMW_i3', 'normal style'),
    'EV2': ('BMW_i3', 'aggressive style'),
    'EV3': ('VW_ID3', 'defensive style'),
    'EV4': ('VW_ID3', 'normal style'),
    'EV5': ('VW_ID3', 'aggressive style'),
    'EV6': ('VW_ID4', 'defensive style'),
    'EV7': ('VW_ID4', 'normal style'),
    'EV8': ('VW_ID4', 'aggressive style'),
    'EV9': ('VW_eUp', 'defensive style'),
    'EV10': ('VW_eUp', 'normal style'),
    'EV11': ('VW_eUp', 'aggressive style'),
    'EV12': ('SUV', 'defensive style'),
    'EV13': ('SUV', 'normal style'),
    'EV14': ('SUV', 'aggressive style')
}

# Battery related features
batttery_simulated = [
    'actualBatteryCapacity(Wh)',
    'SoC(%)',
    'totalEnergyConsumed(Wh)',
    'totalEnergyRegenerated(Wh)',
    'mWh',
]

### MODELS ###
# Models
names = [
    'Decision Tree',
    'Nearest Neighbors',
    'Neural Network',
    'Random Forest'
]

classifiers = [
    DecisionTreeClassifier(random_state=seed),
    KNeighborsClassifier(),
    MLPClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
]

parameters = [
    # DecisionTreeClassifier
    {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(3, 15)  # Narrowed depth range
    },
    # KNeighborsClassifier
    {
        'n_neighbors': list(range(1, 15)),  # Smaller range for n_neighbors
        'weights': ['uniform', 'distance']
    },  
    # MLPClassifier
    {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],  # 'relu' is more commonly used than 'tanh'
        'solver': ['adam']  # 'adam' is generally more efficient and effective
    },
    # RandomForestClassifier
    {
        'criterion': ['gini'],
        'n_estimators': [100, 200]
    }
]
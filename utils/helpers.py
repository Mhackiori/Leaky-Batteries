import numpy as np
import pandas as pd
import os
import random

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn.functional as F

from .params import *



def setSeed(seed=seed):
    """
    Setting the seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def trainGrid(clf, param, X_train, y_train, X_test, y_test):
    """
    GridSearchCV given classifier, parameters and dataset
    """
    grid = GridSearchCV(clf, param, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1
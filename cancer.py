# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 21:03:39 2025

@author: bahad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# warning library
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data_breast.csv")
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)

data = data.rename(columns = {"diagnosis":"target"})

sns.countplot(data["target"])
print(data.target.value_counts())

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

print(len(data))

print(data.head())

print("Data shape ", data.shape)

data.info()

describe = data.describe()

#next: stardization
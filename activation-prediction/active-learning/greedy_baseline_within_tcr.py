import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

from preprocessing import add_activation_thresholds, full_aa_features, get_aa_features
from preprocessing import get_complete_dataset


np.random.seed(42)


start_idx = 0
n_experiments = 10
test_mode = False





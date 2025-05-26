import pandas as pd
import joblib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
from scipy.stats import spearmanr
import joblib
from cv_unstable_correlations.core import run_cv


disease = 'lung'
    
f = open('hyptune%s.txt'%disease, 'w', buffering = 1)

dict_ = {}

for random_seed in range(5):
    for start_year in range(1990, 1996):
        end_year = start_year + 5
        y_list = [joblib.load('y_%s_year%d.joblib'%(disease, year)).reshape(-1) for year in range(start_year, end_year)]
        X_list = [joblib.load('X_%s_year%d.joblib'%(disease, year)) for year in range(start_year, end_year)]
        idx = -1
        
        ts = []
        for year in range(start_year, end_year):
            idx += 1
            print(np.array(X_list[idx]).shape)
            ts.extend([idx]*np.array(X_list[idx]).shape[0])
            
        all_X = np.concatenate(X_list, axis = 0)
        all_y = np.concatenate(y_list, axis = 0)
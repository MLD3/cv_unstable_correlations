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


        for currC in [1e5, 1e4, 1e3, 1e2, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            rand_cv, block_cv, prop_cv = run_cv(all_X, all_y, ts, log_loss, model_kwargs={'n_jobs':3,'solver':'saga', 'penalty':"l1",  'verbose':True, 'random_state':random_seed, 'C':currC})

            print(start_year, currC, 'final')
            curr_model = LogisticRegression(n_jobs = 3, C = currC, solver = 'saga', penalty="l1", random_state=random_seed, verbose = True)
            curr_model.fit(np.array(all_X), all_y)  

            for test_year in range(2000, 2010):
                test_X = joblib.load('X_%s_year%d.joblib'%(disease, test_year))
                test_y = joblib.load('y_%s_year%d.joblib'%(disease, test_year)).reshape(-1)
                pred_test = curr_model.predict_proba(test_X)[:, 1]
                test_loss = log_loss(test_y, pred_test)
                dict_[(random_seed, start_year, currC, test_year)] = [pred_test, test_y]
                f.write('random_seed:%d, start_year:%d, test_year:%d, currC:%0.10f\n'%(random_seed, start_year, test_year, currC))
                f.write('BLOCK: %0.6f\n'%(block_cv))
                f.write('RAND: %0.6f\n'%(rand_cv))
                f.write('PROP: %0.6f\n'%(prop_cv))
                f.write('TEST: %0.6f\n'%(test_loss))
                f.write('\n')

f.close()

joblib.dump(dict_, 'dict_loss.joblib')
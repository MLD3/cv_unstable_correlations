import pandas as pd
import joblib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from cv_unstable_correlations.core import run_cv

count = 0

f = open('semi_synthetic_auroc.txt', 'w', buffering = 1)

careunits_to_use = pickle.load(open('updatedtimewindow_careunits.pkl', 'rb'))
careunit_to_use = 1734

old_train_X1 = pickle.load(open('updatedtimewindow_old_train_X1.pkl', 'rb'))
old_train_X2 = pickle.load(open('updatedtimewindow_old_train_X2.pkl', 'rb'))
old_train_X3 = pickle.load(open('updatedtimewindow_old_train_X3.pkl', 'rb'))

train_X1 = pickle.load(open('updatedtimewindow_train_X1.pkl', 'rb'))
train_X2 = pickle.load(open('updatedtimewindow_train_X2.pkl', 'rb'))
train_X3 = pickle.load(open('updatedtimewindow_train_X3.pkl', 'rb'))

train_y1 = pickle.load(open('updatedtimewindow_train_y1.pkl', 'rb'))
train_y2 = pickle.load(open('updatedtimewindow_train_y2.pkl', 'rb'))
train_y3 = pickle.load(open('updatedtimewindow_train_y3.pkl', 'rb'))

train_y1 = np.array(train_y1) > 3
train_y2 = np.array(train_y2) > 3
train_y3 = np.array(train_y3) > 3

all_X = np.concatenate((np.array(train_X1), np.array(train_X2), np.array(train_X3)), axis = 0)
all_y = np.concatenate((np.array(train_y1), np.array(train_y2), np.array(train_y3)), axis = 0)

old_test_X = np.array(pickle.load(open('updatedtimewindow_old_test_X.pkl', 'rb')))
test_X = np.array(pickle.load(open('updatedtimewindow_test_X.pkl', 'rb')))
test_y = np.array(pickle.load(open('updatedtimewindow_test_y.pkl', 'rb'))) > 3


test_X_LOShigh_careunithigh = test_X[np.logical_and(test_y > 0 , old_test_X[:, careunit_to_use] > 0)]
test_y_LOShigh_careunithigh = test_y[np.logical_and(test_y > 0 , old_test_X[:, careunit_to_use] > 0)]
test_X_LOSlow_careunitlow = test_X[np.logical_and(test_y == 0 , old_test_X[:, careunit_to_use] == 0)]
test_y_LOSlow_careunitlow = test_y[np.logical_and(test_y == 0 , old_test_X[:, careunit_to_use] == 0)]
test_X_LOSlow_careunithigh = test_X[np.logical_and(test_y == 0, old_test_X[:, careunit_to_use] > 0)]
test_y_LOSlow_careunithigh = test_y[np.logical_and(test_y == 0 , old_test_X[:, careunit_to_use] > 0)]

test_X_LOShigh_careunitlow = test_X[np.logical_and(test_y > 0, old_test_X[:, careunit_to_use] == 0)]
test_y_LOShigh_careunitlow = test_y[np.logical_and(test_y > 0 , old_test_X[:, careunit_to_use] == 0)]

new_test_X = []
new_test_y = []

print('test_X')
count = 0
whilecounter = 0
while whilecounter < 50000:
    if count == len(test_X_LOSlow_careunithigh):
        count = 0
    new_test_X.append(test_X_LOSlow_careunithigh[count])
    new_test_y.append(test_y_LOSlow_careunithigh[count])
    count += 1
    whilecounter += 1

count = 0
whilecounter = 0
while whilecounter < 50000: 
    if count == len(test_X_LOShigh_careunitlow):
        count = 0
    new_test_X.append(test_X_LOShigh_careunitlow[count])
    new_test_y.append(test_y_LOShigh_careunitlow[count])
    count += 1
    whilecounter += 1


whilecounter = 0
count = 0
while whilecounter < 50000:
    if count == len(test_X_LOShigh_careunithigh):
        count = 0
    new_test_X.append(test_X_LOShigh_careunithigh[count])
    new_test_y.append(test_y_LOShigh_careunithigh[count])
    count += 1
    whilecounter += 1

count = 0
whilecounter = 0
while whilecounter < 50000:
    if count == len(test_X_LOSlow_careunitlow):
        count = 0
    new_test_X.append(test_X_LOSlow_careunitlow[count])
    new_test_y.append(test_y_LOSlow_careunitlow[count])
    count += 1
    whilecounter += 1
                

print('corr', spearmanr(np.array(new_test_X)[:, careunit_to_use], np.array(new_test_y))[0])


X1_LOShigh_careunithigh = train_X1[np.logical_and(train_y1 > 0 , old_train_X1[:, careunit_to_use] > 0)]
y1_LOShigh_careunithigh = train_y1[np.logical_and(train_y1 > 0 , old_train_X1[:, careunit_to_use] > 0)]

X2_LOShigh_careunithigh = train_X2[np.logical_and(train_y2 > 0 , old_train_X2[:, careunit_to_use] > 0)]
y2_LOShigh_careunithigh = train_y2[np.logical_and(train_y2 > 0 , old_train_X2[:, careunit_to_use] > 0)]

X3_LOShigh_careunithigh = train_X3[np.logical_and(train_y3 > 0 , old_train_X3[:, careunit_to_use] > 0)]
y3_LOShigh_careunithigh = train_y3[np.logical_and(train_y3 > 0 , old_train_X3[:, careunit_to_use] > 0)]


X1_LOSlow_careunitlow = train_X1[np.logical_and(train_y1 == 0 , old_train_X1[:, careunit_to_use] == 0)]
y1_LOSlow_careunitlow = train_y1[np.logical_and(train_y1 == 0 , old_train_X1[:, careunit_to_use] == 0)]

X2_LOSlow_careunitlow = train_X2[np.logical_and(train_y2 == 0 , old_train_X2[:, careunit_to_use] == 0)]
y2_LOSlow_careunitlow = train_y2[np.logical_and(train_y2 == 0 , old_train_X2[:, careunit_to_use] == 0)]

X3_LOSlow_careunitlow = train_X3[np.logical_and(train_y3 == 0 , old_train_X3[:, careunit_to_use] == 0)]
y3_LOSlow_careunitlow = train_y3[np.logical_and(train_y3 == 0 , old_train_X3[:, careunit_to_use] == 0)]

X1_LOShigh_careunitlow = train_X1[np.logical_and(train_y1 > 0 , old_train_X1[:, careunit_to_use] == 0)]
y1_LOShigh_careunitlow = train_y1[np.logical_and(train_y1 > 0 , old_train_X1[:, careunit_to_use] == 0)]

X2_LOShigh_careunitlow = train_X2[np.logical_and(train_y2 > 0 , old_train_X2[:, careunit_to_use] == 0)]
y2_LOShigh_careunitlow = train_y2[np.logical_and(train_y2 > 0 , old_train_X2[:, careunit_to_use] == 0)]

X3_LOShigh_careunitlow = train_X3[np.logical_and(train_y3 > 0 , old_train_X3[:, careunit_to_use] == 0)]
y3_LOShigh_careunitlow = train_y3[np.logical_and(train_y3 > 0 , old_train_X3[:, careunit_to_use] == 0)]

X1_LOSlow_careunithigh = train_X1[np.logical_and(train_y1 == 0, old_train_X1[:, careunit_to_use] > 0)]
y1_LOSlow_careunithigh = train_y1[np.logical_and(train_y1 == 0 , old_train_X1[:, careunit_to_use] > 0)]

X2_LOSlow_careunithigh = train_X2[np.logical_and(train_y2 == 0 , old_train_X2[:, careunit_to_use] > 0)]
y2_LOSlow_careunithigh = train_y2[np.logical_and(train_y2 == 0 , old_train_X2[:, careunit_to_use] > 0)]

X3_LOSlow_careunithigh = train_X3[np.logical_and(train_y3 == 0 , old_train_X3[:, careunit_to_use] > 0)]
y3_LOSlow_careunithigh = train_y3[np.logical_and(train_y3 == 0 , old_train_X3[:, careunit_to_use] > 0)]


old_stable_features = np.array(list(range(train_X1.shape[1])))

stable_features = []
unstable_features = []
for ft in old_stable_features:
    if ft not in careunits_to_use:
        stable_features.append(ft)
        unstable_features.append(ft)

unstable_features.append(careunit_to_use)
stable_features = np.array(stable_features)
unstable_features = np.array(unstable_features)


print('CREATE NEW DATASET')

COUNT = -1
arr1 = np.array([0, 5000, 10000, 23000, 34000, 50000])
arr2 = 100000 - arr1

for CHOICE1 in range(len(arr1)):
    for CHOICE2 in range(len(arr1)):
        for CHOICE3 in range(len(arr1)):
            COUNT += 1
            new_train_X1 = []
            new_train_y1 = []

            new_train_X2 = []
            new_train_y2 = []

            new_train_X3 = []
            new_train_y3 = []

            print(CHOICE3, 'X1')
            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE1]:
                if count == len(X1_LOSlow_careunithigh):
                    count = 0
                new_train_X1.append(X1_LOSlow_careunithigh[count])
                new_train_y1.append(y1_LOSlow_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE1]: 
                if count == len(X1_LOShigh_careunitlow):
                    count = 0
                new_train_X1.append(X1_LOShigh_careunitlow[count])
                new_train_y1.append(y1_LOShigh_careunitlow[count])
                count += 1
                whilecounter += 1


            whilecounter = 0
            count = 0
            while whilecounter < arr2[CHOICE1]:
                if count == len(X1_LOShigh_careunithigh):
                    count = 0
                new_train_X1.append(X1_LOShigh_careunithigh[count])
                new_train_y1.append(y1_LOShigh_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr2[CHOICE1]:
                if count == len(X1_LOSlow_careunitlow):
                    count = 0
                new_train_X1.append(X1_LOSlow_careunitlow[count])
                new_train_y1.append(y1_LOSlow_careunitlow[count])
                count += 1
                whilecounter += 1

            print(CHOICE3, 'X2')
            # X2
            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE2]:
                if count == len(X2_LOSlow_careunithigh):
                    count = 0
                new_train_X2.append(X2_LOSlow_careunithigh[count])
                new_train_y2.append(y2_LOSlow_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE2]:
                if count == len(X2_LOShigh_careunitlow):
                    count = 0
                new_train_X2.append(X2_LOShigh_careunitlow[count])
                new_train_y2.append(y2_LOShigh_careunitlow[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr2[CHOICE2]:
                if count == len(X2_LOShigh_careunithigh):
                    count = 0
                new_train_X2.append(X2_LOShigh_careunithigh[count])
                new_train_y2.append(y2_LOShigh_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr2[CHOICE2]:
                if count == len(X2_LOSlow_careunitlow):
                    count = 0
                new_train_X2.append(X2_LOSlow_careunitlow[count])
                new_train_y2.append(y2_LOSlow_careunitlow[count])
                count += 1
                whilecounter += 1


            print(CHOICE3, 'X3')
            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE3]:
                if count == len(X3_LOSlow_careunithigh):
                    count = 0
                new_train_X3.append(X3_LOSlow_careunithigh[count])
                new_train_y3.append(y3_LOSlow_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr1[CHOICE3]:
                if count == len(X3_LOShigh_careunitlow):
                    count = 0
                new_train_X3.append(X3_LOShigh_careunitlow[count])
                new_train_y3.append(y3_LOShigh_careunitlow[count])
                count += 1
                whilecounter += 1


            count = 0
            whilecounter = 0
            while whilecounter < arr2[CHOICE3]:
                if count == len(X3_LOShigh_careunithigh):
                    count = 0
                new_train_X3.append(X3_LOShigh_careunithigh[count])
                new_train_y3.append(y3_LOShigh_careunithigh[count])
                count += 1
                whilecounter += 1

            count = 0
            whilecounter = 0
            while whilecounter < arr2[CHOICE3]:
                if count == len(X3_LOSlow_careunitlow):
                    count = 0
                new_train_X3.append(X3_LOSlow_careunitlow[count])
                new_train_y3.append(y3_LOSlow_careunitlow[count])
                count += 1
                whilecounter += 1

            print(len(new_train_y1), len(new_train_y2), len(new_train_y3))

            new_train_y1 = np.array(new_train_y1) 
            new_train_y2 = np.array(new_train_y2) 
            new_train_y3 = np.array(new_train_y3) 

            new_stable_train_X1 = np.array(new_train_X1)[:, stable_features]
            new_stable_train_X2 = np.array(new_train_X2)[:, stable_features]
            new_stable_train_X3 = np.array(new_train_X3)[:, stable_features]

            new_unstable_train_X1 = np.array(new_train_X1)[:, unstable_features]
            new_unstable_train_X2 = np.array(new_train_X2)[:, unstable_features]
            new_unstable_train_X3 = np.array(new_train_X3)[:, unstable_features]

            ts = []
            for idx, curr_y in enumerate([new_train_y1, new_train_y2, new_train_y3]):
                ts.extend([idx]*len(curr_y))
            
            all_unstable_X = np.concatenate((np.array(new_unstable_train_X1), np.array(new_unstable_train_X2), np.array(new_unstable_train_X3)), axis = 0)
            all_stable_X = np.concatenate((np.array(new_stable_train_X1), np.array(new_stable_train_X2), np.array(new_stable_train_X3)), axis = 0)
            all_y = np.concatenate((np.array(new_train_y1), np.array(new_train_y2), np.array(new_train_y3)), axis = 0)


            stable_rand_cv, stable_block_cv, stable_prop_cv = run_cv(X_rand_stable, y_rand_stable, ts, roc_auc_score)
            unstable_rand_cv, unstable_block_cv, unstable_prop_cv = run_cv(X_rand_stable, y_rand_stable, ts, roc_auc_score)

            stable = LogisticRegression(random_state = 0)
            stable.fit(all_stable_X, all_y)
            stable_pred_test = stable.predict_proba(np.array(new_test_X)[:, stable_features])[:, 1]
            stable_test_auroc = roc_auc_score(new_test_y, stable_pred_test)

            unstable = LogisticRegression(random_state = 0)
            unstable.fit(np.array(all_unstable_X), all_y)
            unstable_pred_test = unstable.predict_proba(np.array(new_test_X)[:, unstable_features])[:, 1]
            unstable_test_auroc = roc_auc_score(new_test_y, unstable_pred_test)

            print(COUNT)
            print(stable_block_cv, unstable_block_cv)
            print(stable_rand_cv, unstable_rand_cv)
            print(stable_prop_cv, unstable_prop_cv)
            print(stable_test_auroc, unstable_test_auroc)
            print()


            A = spearmanr(all_unstable_X[:, all_unstable_X.shape[1] - 1], all_y)[0]
            r1 = spearmanr(new_unstable_train_X1[:, new_unstable_train_X1.shape[1] - 1].reshape(-1), new_train_y1)[0]
            r2 = spearmanr(new_unstable_train_X2[:, new_unstable_train_X2.shape[1] - 1].reshape(-1), new_train_y2)[0]
            r3 = spearmanr(new_unstable_train_X3[:, new_unstable_train_X3.shape[1] - 1].reshape(-1), new_train_y3)[0]
            V = np.var([r1, r2, r3])

            print(COUNT, 'A:%0.3f, V:%0.3f, max_ft:%d'%(A, V, np.argmax(unstable.coef_.reshape(-1))))
            print(stable_block_cv, unstable_block_cv)
            print(stable_rand_cv, unstable_rand_cv)
            print(stable_prop_cv, unstable_prop_cv)
            print(stable_test_auroc, unstable_test_auroc)
            print()

            f.write('%d, A:%0.3f, V:%0.3f, max_ft:%d\n'%(COUNT, A, V, np.argmax(unstable.coef_.reshape(-1))))
            f.write('BLOCK: stable:%0.3f, unstable%0.3f\n'%(stable_block_cv, unstable_block_cv))
            f.write('RAND: stable:%0.3f, unstable%0.3f\n'%(stable_rand_cv, unstable_rand_cv))
            f.write('PROP: stable:%0.3f, unstable%0.3f\n'%(stable_prop_cv, unstable_prop_cv))
            f.write('TEST: stable:%0.3f, unstable%0.3f\n'%(stable_test_auroc, unstable_test_auroc))
            f.write('\n')


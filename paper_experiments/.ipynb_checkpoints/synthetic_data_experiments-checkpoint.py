import numpy as np
from sklearn.metrics import mean_squared_error as loss
from cv_unstable_correlations.core import run_cv
from sklearn.linear_model import LinearRegression as model

N = 10000
  
dict_ = {'A':[], 'V':[]}
dict_['$\\frac{a}{b}$'] = []
dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'] = []
dict_['Method'] = []
dict_['Resulting Model MSE'] = []

for A in [0.05, 0.25, 0.5, 0.75, 0.95]:
    for amin_div in [10, 5, 2, 1.1]:
        simcount = 0
        while simcount < 1000:
            T = 10
            p = int(N/(T-1))

            maxp = 1
            V = (A*maxp-(A**2))*np.random.rand()

            alpha = (-A**3 +(A**2)*maxp - A*V)/(maxp*V)
            beta = (A - maxp)*(A**2 - A*maxp + V)/(maxp*V)

            pks = list(np.random.beta(alpha, beta, T - 1)*maxp)

            newA = np.mean(pks)
            newV = np.std(pks)**2

            pks.append(0)


            x1 = []
            x2 = []
            y = []

            b = np.random.rand()*15
            
            amax = np.sqrt(newA*(2-newA))*b
            amin = amax/amin_div
            a = (amax - amin)*np.random.rand() + amin

            for t in range(T):
                curr_s_vals = np.random.normal(0, 1, N)
                curr_u_vals = np.random.normal(0, 1, N)

                U = np.random.permutation(N)[:int(pks[t]*N)]
                for n in range(N):
                    curr_y = a*curr_s_vals[n] + b*curr_u_vals[n]
                    if n not in U:
                        curr_y = a*curr_s_vals[n]
                    y.append(curr_y)
                x1.extend(curr_s_vals)
                x2.extend(curr_u_vals)

            x = np.concatenate((np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1)), 1)
            y = np.array(y)

            X_train = x[:(T - 1)*N]
            y_train = y[:(T - 1)*N]
            X_test = x[(T - 1)*N:]
            y_test = y[(T - 1)*N:]

            ts = []
            for b in range(T - 1):
                ts.extend([b]*N)

            stable_rand, stable_block, stable_diff = run_cv(X_train[:, 0], y_train, ts, loss, model_type='linreg', model_kwargs={'fit_intercept':False, 'positive':True})
            unstable_rand, unstable_block, unstable_diff = run_cv(X_train[:, 1], y_train, ts, loss, model_type='linreg', model_kwargs={'fit_intercept':False, 'positive':True})

            V_cond = (((newA**2)*(b**2) - a**2)*((T-2)**2))/((b**2)*(2*(T - 1) - 1))
            
            clf = model(fit_intercept = False, positive=True).fit(X_train[:, [0]], y_train)
            actual_a = clf.coef_[0]
            y_pred = clf.predict(X_test[:][:, [0]]) 
            choose_0_loss = loss(y_test, y_pred)

            clf = model(fit_intercept = False, positive=True).fit(X_train[:, [1]], y_train)
            actual_Ab = clf.coef_[0]
            y_pred = clf.predict(X_test[:][:, [1]]) 
            choose_1_loss = loss(y_test, y_pred)
            
            if choose_1_loss - choose_0_loss > 0.1:
                simcount += 1
            
                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Random\nCross\nValidation')
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin([stable_rand, unstable_rand])])

                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Block\nCross\nValidation')
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin([stable_block, unstable_block])])

                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Proposed\nApproach')
                diff_choices = np.array([stable_diff, unstable_diff])
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin(diff_choices)])


                # print(choose_0_loss, choose_1_loss)
                # print()

                print(simcount, A, amin_div)

                if choose_0_loss > choose_1_loss:
                    print('ERROR')

            
for V in np.linspace(0.01, 0.24, 5):
    for amin_div in [10, 5, 2, 1.1]:
        simcount = 0
        while simcount < 1000:
            T = 10
            p = int(N/(T-1))

            # 1/2 (1 - sqrt(1 - 4 V))<=A<=1/2 (sqrt(1 - 4 V) + 1) and V<1/4
            lower_bound = (1/2)*(1 - np.sqrt(1 - 4*V))
            upper_bound = (1/2)*(np.sqrt(1 - 4*V) + 1)
            A = np.random.rand()*(upper_bound - lower_bound) + lower_bound


            alpha = (-A**3 +(A**2)*maxp - A*V)/(maxp*V)
            beta = (A - maxp)*(A**2 - A*maxp + V)/(maxp*V)

            pks = list(np.random.beta(alpha, beta, T - 1))

            newA = np.mean(pks)
            newV = np.std(pks)**2

            pks.append(0)


            x1 = []
            x2 = []
            y = []

            b = np.random.rand()*15
            amax = np.sqrt(newA*(2-newA))*b
            amin = amax/amin_div
            a = (amax - amin)*np.random.rand() + amin

            for t in range(T):
                curr_s_vals = np.random.normal(0, 1, N)
                curr_u_vals = np.random.normal(0, 1, N)

                U = np.random.permutation(N)[:int(pks[t]*N)]
                for n in range(N):
                    curr_y = a*curr_s_vals[n] + b*curr_u_vals[n]
                    if n not in U:
                        curr_y = a*curr_s_vals[n]
                    y.append(curr_y)
                x1.extend(curr_s_vals)
                x2.extend(curr_u_vals)



            x = np.concatenate((np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1)), 1)
            y = np.array(y)

            X_train = x[:(T - 1)*N]
            y_train = y[:(T - 1)*N]
            X_test = x[(T - 1)*N:]
            y_test = y[(T - 1)*N:]

            ts = []
            for b in range(T - 1):
                ts.extend([b]*N)

            stable_rand, stable_block, stable_diff = run_cv(X_train[:, 0], y_train, ts, loss, model_type='linreg', model_kwargs={'fit_intercept':False, 'positive':True})
            unstable_rand, unstable_block, unstable_diff = run_cv(X_train[:, 1], y_train, ts, loss, model_type='linreg', model_kwargs={'fit_intercept':False, 'positive':True})

            V_cond = (((newA**2)*(b**2) - a**2)*((T-2)**2))/((b**2)*(2*(T - 1) - 1))
            
            clf = model(fit_intercept = False, positive=True).fit(X_train[:, [0]], y_train)
            actual_a = clf.coef_[0]
            y_pred = clf.predict(X_test[:][:, [0]]) 
            choose_0_loss = loss(y_test, y_pred)

            clf = model(fit_intercept = False, positive=True).fit(X_train[:, [1]], y_train)
            actual_Ab = clf.coef_[0]
            y_pred = clf.predict(X_test[:][:, [1]]) 
            choose_1_loss = loss(y_test, y_pred)

            if choose_1_loss - choose_0_loss > 0.1:
                simcount += 1
            
                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Random\nCross\nValidation')
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin([stable_rand, unstable_rand])])

                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Block\nCross\nValidation')
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin([stable_block, unstable_block])])

                dict_['A'].append(newA)
                dict_['V'].append(newV)
                dict_['$\\frac{a}{b}$'].append(a/b)
                dict_['$$\\frac{(A^2b^2 - a^2)(K-1)^2}{b^2(2K-1)}$$'].append(V_cond)
                dict_['Method'].append('Proposed\nApproach')
                diff_choices = np.array([stable_diff, unstable_diff])
                dict_['Resulting Model MSE'].append([choose_0_loss, choose_1_loss][np.argmin(diff_choices)])


                # print(choose_0_loss, choose_1_loss)
                # print()

                print(simcount, V, amin_div)

                if choose_0_loss > choose_1_loss:
                    print('ERROR')
            

np.save('synthetic_data_results.npy', dict_) 
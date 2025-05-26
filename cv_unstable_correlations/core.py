import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

def cv_logreg(X_array, y_array, metric_fn, model_kwargs=None):    
    if model_kwargs is None:
        model_kwargs = {}
    curr_losses = []
    for i in range(len(X_array)):
        X_heldout = X_array[i]
        y_heldout = y_array[i]
        X_train = np.concatenate([X_array[j] for j in range(len(X_array)) if j != i], axis=0)
        y_train = np.concatenate([y_array[j] for j in range(len(y_array)) if j != i], axis=0)

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_heldout.ndim == 1:
            X_heldout = X_heldout.reshape(-1, 1)
        
        clf = LogisticRegression(**model_kwargs).fit(X_train, y_train)
        y_pred = clf.predict_proba(X_heldout)[:, 1]
        loss = metric_fn(y_heldout, y_pred)
        curr_losses.append(loss)
    return np.mean(curr_losses)

def cv_linreg(X_array, y_array, metric_fn, model_kwargs=None):    
    if model_kwargs is None:
        model_kwargs = {}
    curr_losses = []
    for i in range(len(X_array)):
        X_heldout = X_array[i]
        y_heldout = y_array[i]
        X_train = np.concatenate([X_array[j] for j in range(len(X_array)) if j != i], axis=0)
        y_train = np.concatenate([y_array[j] for j in range(len(y_array)) if j != i], axis=0)

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_heldout.ndim == 1:
            X_heldout = X_heldout.reshape(-1, 1)

        clf = LinearRegression(**model_kwargs).fit(X_train, y_train)
        y_pred = clf.predict(X_heldout)
        loss = metric_fn(y_heldout, y_pred)
        curr_losses.append(loss)
    return np.mean(curr_losses)


def run_cv(X, y, ts, metric_fn, model_type='logreg', model_kwargs=None):
    """
    Compare instability between block CV and random CV over time-indexed data.

    Parameters:
    - X: feature matrix of shape (n_samples, d)
    - y: targets (n_samples,)
    - ts: time index (same length as y), values should correspond to K time blocks
    - metric_fn: scoring function (e.g., log-loss, MAE)
    - model_type: 'logreg' or 'linreg'

    Returns:
    - instability_score: float
    """
    
    K = len(list(set(ts)))
    N = len(y) // K

    # Random CV splits

    rand_idxs = np.random.permutation(len(y))
    print('random_state' in model_kwargs.keys(), model_kwargs['random_state'])
    if 'random_state' in model_kwargs.keys():
        rand_idxs = np.random.RandomState(seed=model_kwargs['random_state']).permutation(len(y))
    
    X_rand = [X[rand_idxs[b * N:(b + 1) * N]] for b in range(K)]
    y_rand = [y[rand_idxs[b * N:(b + 1) * N]] for b in range(K)]

    # Block CV splits based on time values
    t_ordered = sorted(set(ts))
    
    X_block = [X[np.array(ts) == ti] for ti in t_ordered]
    y_block = [y[np.array(ts) == ti] for ti in t_ordered]
    

    if model_type == 'logreg':
        rand_score = cv_logreg(X_rand, y_rand, metric_fn, model_kwargs)
        block_score = cv_logreg(X_block, y_block, metric_fn, model_kwargs)
    elif model_type == 'linreg':
        rand_score = cv_linreg(X_rand, y_rand, metric_fn, model_kwargs)
        block_score = cv_linreg(X_block, y_block, metric_fn, model_kwargs)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    return rand_score, block_score, abs(block_score - rand_score)

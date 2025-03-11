import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from config import *

def train_value(solved_duals, feature_vectors, method=PREDICTION_METHOD, n_neighbors=N_NEIGHBORS, seed=SEED):
    """
    Train a model (Linear Regression or KNN) to predict dual variables.

    Args:
        solved_duals (dict): Solved dual values {scenario: (mu, nu)} for selected scenarios.
        feature_vectors (dict): Feature vectors for all scenarios.
        method (str): The prediction method, either "regression" or "knn".
        n_neighbors (int): Number of neighbors for KNN.

    Returns:
        model_mu, model_nu: Trained models for mu and nu.
    """
    np.random.seed(seed)  # Ensuring consistency

    selected_scenarios = list(solved_duals.keys())

    # Prepare training data
    X_train = np.array([feature_vectors[s] for s in selected_scenarios])  
    y_mu_train = np.array([np.array(list(solved_duals[s][0].values()), dtype=float) for s in selected_scenarios])
    y_nu_train = np.array([np.array(list(solved_duals[s][1].values()), dtype=float) for s in selected_scenarios])

    if method == "regression":
        model_mu = LinearRegression().fit(X_train, y_mu_train)
        model_nu = LinearRegression().fit(X_train, y_nu_train)
    elif method == "knn":
        model_mu = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_mu_train)
        model_nu = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_nu_train)
    else:
        raise ValueError("Invalid method. Choose 'regression' or 'knn'.")

    return model_mu, model_nu

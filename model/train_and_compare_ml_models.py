import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Load dataset
Plastic = pd.read_excel('train_random_all_1_6.xlsx')

X = Plastic.values[:, 0:6]
Y = Plastic.values[:, 6:12]

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Define models and hyperparameter search spaces
models = {
    'DT': {
        'model': DecisionTreeRegressor(random_state=42),
        'param_dist': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'RF': {
        'model': RandomForestRegressor(random_state=42),
        'param_dist': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'SVR': {
        'model': MultiOutputRegressor(SVR()),
        'param_dist': {
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__gamma': ['scale', 'auto'],
            'estimator__kernel': ['linear', 'rbf']
        }
    },
    'GBR': {
        'model': GradientBoostingRegressor(random_state=42),
        'param_dist': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}

# Train, tune, and evaluate models
for model_name, model_info in models.items():
    model = model_info['model']
    param_dist = model_info['param_dist']

    start_time = time.time()

    random_search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=5,
        cv=5,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    runtime = time.time() - start_time

    print(f'{model_name} - Best Hyperparameters: {random_search.best_params_}')
    print(f'{model_name} - Train R2: {r2_train:.4f}')
    print(f'{model_name} - Train RMSE: {rmse_train:.4f}')
    print(f'{model_name} - Train MAE: {mae_train:.4f}')
    print(f'{model_name} - Test R2: {r2:.4f}')
    print(f'{model_name} - Test RMSE: {rmse:.4f}')
    print(f'{model_name} - Test MAE: {mae:.4f}')
    print(f'Runtime: {runtime:.2f} s')
    print('-' * 50)

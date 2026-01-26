import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load dataset
Plastic = pd.read_excel('train random all 1_6.xlsx')
X = Plastic.values[:, 0:6]
Y = Plastic.values[:, 6:12]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 2. Train baseline Random Forest model
model = RandomForestRegressor(
    n_estimators=67,
    max_depth=27,
    random_state=42
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Baseline performance:")
print("Train R2:", r2_score(y_train, y_train_pred))
print("Test  R2:", r2_score(y_test, y_test_pred))

# 3. Learning curve by training data fraction
def mixture_complexity(y):
    return np.count_nonzero(y)

complexity_levels = np.array([mixture_complexity(y) for y in y_train])

strata_indices = defaultdict(list)
for idx, c in enumerate(complexity_levels):
    strata_indices[c].append(idx)

rng = np.random.RandomState(42)
train_fractions = np.linspace(0.1, 1.0, 10)

train_r2_size = []
test_r2_size = []
train_rmse_size = []
test_rmse_size = []

for frac in train_fractions:
    selected_indices = []
    for c, indices in strata_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n_select = int(frac * len(indices))
        if n_select < 1:
            continue
        selected_indices.extend(indices[:n_select])
    selected_indices = np.array(selected_indices)

    X_sub = X_train[selected_indices]
    y_sub = y_train[selected_indices]

    rf = RandomForestRegressor(n_estimators=67, max_depth=27, random_state=42)
    rf.fit(X_sub, y_sub)

    y_sub_pred = rf.predict(X_sub)
    y_test_pred = rf.predict(X_test)

    train_r2_size.append(r2_score(y_sub, y_sub_pred))
    test_r2_size.append(r2_score(y_test, y_test_pred))
    train_rmse_size.append(np.sqrt(mean_squared_error(y_sub, y_sub_pred)))
    test_rmse_size.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Plot learning curve by data fraction
fig, ax1 = plt.subplots(figsize=(8, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Fraction of training data', fontsize=12)
ax1.set_ylabel('R²', color=color1, fontsize=12)
ax1.plot(train_fractions, train_r2_size, marker='o', color=color1, label='Training R²')
ax1.plot(train_fractions, test_r2_size, marker='s', color=color1, linestyle='--', label='Test R²')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('RMSE', color=color2, fontsize=12)
ax2.plot(train_fractions, train_rmse_size, marker='^', color=color2, label='Training RMSE')
ax2.plot(train_fractions, test_rmse_size, marker='v', color=color2, linestyle='--', label='Test RMSE')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()

# Export learning curve data (fraction) to Excel
df_fraction = pd.DataFrame({
    'Training_Data_Fraction': train_fractions,
    'Training_R2': train_r2_size,
    'Test_R2': test_r2_size,
    'Training_RMSE': train_rmse_size,
    'Test_RMSE': test_rmse_size
})
df_fraction.to_excel('learning_curve_fraction_data.xlsx', index=False)
print("Learning curve data by fraction exported successfully.")

# 4. Learning curve by number of trees
n_estimators_list = [10, 20, 30, 40, 50, 67, 80, 100, 150, 300]
train_r2_trees = []
test_r2_trees = []
train_rmse_trees = []
test_rmse_trees = []

for n_trees in n_estimators_list:
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=27, random_state=42)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2_trees.append(r2_score(y_train, y_train_pred))
    test_r2_trees.append(r2_score(y_test, y_test_pred))
    train_rmse_trees.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse_trees.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Plot learning curve by number of trees
fig, ax1 = plt.subplots(figsize=(8, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Number of trees', fontsize=12)
ax1.set_ylabel('R²', color=color1, fontsize=12)
ax1.plot(n_estimators_list, train_r2_trees, marker='o', color=color1, label='Training R²')
ax1.plot(n_estimators_list, test_r2_trees, marker='s', color=color1, linestyle='--', label='Test R²')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('RMSE', color=color2, fontsize=12)
ax2.plot(n_estimators_list, train_rmse_trees, marker='^', color=color2, label='Training RMSE')
ax2.plot(n_estimators_list, test_rmse_trees, marker='v', color=color2, linestyle='--', label='Test RMSE')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()

# Export learning curve data (trees) to Excel
df_trees = pd.DataFrame({
    'Number_of_Trees': n_estimators_list,
    'Training_R2': train_r2_trees,
    'Test_R2': test_r2_trees,
    'Training_RMSE': train_rmse_trees,
    'Test_RMSE': test_rmse_trees
})
df_trees.to_excel('learning_curve_trees_data.xlsx', index=False)
print("Learning curve data by number of trees exported successfully.")


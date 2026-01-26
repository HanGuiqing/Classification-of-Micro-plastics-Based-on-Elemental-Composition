import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import shap
import time

# Global font settings
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 24
plt.rcParams.update({'font.size': 18})

target_labels = ["PET", "PE&PP", "PVC", "PS", "PA", "PC"]
feature_names = ['C', 'H', 'O', 'N', 'H/C', 'O/C']
target_names = target_labels

# Parity plot for all data
fig, ax = plt.subplots(figsize=(6, 6), dpi=700)
plt.xlabel('Actual value (%)', fontsize=20, weight='bold', family='Times New Roman')
plt.ylabel('Predicted value (%)', fontsize=20, weight='bold', family='Times New Roman')
plt.xticks(fontsize=20, weight='bold', family='Times New Roman')
plt.yticks(fontsize=20, weight='bold', family='Times New Roman')

x = np.linspace(0, 80, 20)
y = x
plt.plot(x, y, color='darkred', linewidth=2, linestyle='--', label='Bisect', alpha=1)

sns.set(color_codes=True)
plt.scatter(y_train, y_train_pred, color='dodgerblue', marker='>', label='Training', s=1, edgecolor='dodgerblue', alpha=0.6)
plt.scatter(y_test, y_test_pred, color='red', marker='o', label='Testing', s=1, edgecolor='red', alpha=0.1)

plt.legend(loc='upper center', frameon=False, prop={"weight":"bold","family": "Times New Roman", "size": 14})
plt.text(0, 75,'Train\nR${^2}$: 0.99\nRMSE:0.79\nMAE:0.48', fontsize=20, weight='bold', color='red', family='Times New Roman')
plt.text(70, 0,'Test\nR${^2}$: 0.98\nRMSE:2.04\nMAE:1.28', fontsize=20, weight='bold', color='dodgerblue', family='Times New Roman')
plt.text(85, 60,'RF', fontsize=20, weight='bold', family='Times New Roman')

plt.rcParams['axes.facecolor'] = 'white'
ax = plt.gca()
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

plt.show()

# Metrics for each output
for i in range(Y.shape[1]):
    train_r2_y = r2_score(y_train[:, i], y_train_pred[:, i])
    test_r2_y = r2_score(y_test[:, i], y_test_pred[:, i])
    train_rmse_y = mean_squared_error(y_train[:, i], y_train_pred[:, i])**0.5
    test_rmse_y = mean_squared_error(y_test[:, i], y_test_pred[:, i])**0.5
    train_mae_y = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
    test_mae_y = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
    print(f'Y{i+1} - Train R2: {np.around(train_r2_y, 4)}, Test R2: {np.around(test_r2_y, 4)}, '
          f'Train RMSE: {np.around(train_rmse_y, 4)}, Test RMSE: {np.around(test_rmse_y, 4)}, '
          f'Train MAE: {np.around(train_mae_y, 4)}, Test MAE: {np.around(test_mae_y, 4)}')

# Predictions for each output
y_train_pred_all = model.predict(X_train)
y_test_pred_all = model.predict(X_test)

# Individual parity plots
for i, col_name in enumerate(target_labels):
    y_train_i = y_train[:, i]
    y_test_i = y_test[:, i]
    y_train_pred_i = y_train_pred_all[:, i]
    y_test_pred_i = y_test_pred_all[:, i]

    r2_train = r2_score(y_train_i, y_train_pred_i)
    r2_test = r2_score(y_test_i, y_test_pred_i)
    rmse_train = mean_squared_error(y_train_i, y_train_pred_i) ** 0.5
    rmse_test = mean_squared_error(y_test_i, y_test_pred_i) ** 0.5
    mae_train = mean_absolute_error(y_train_i, y_train_pred_i)
    mae_test = mean_absolute_error(y_test_i, y_test_pred_i)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=700)
    plt.xlabel('Actual value (%)', fontsize=20, weight='bold', family='Times New Roman')
    plt.ylabel('Predicted value (%)', fontsize=20, weight='bold', family='Times New Roman')
    plt.xticks(fontsize=20, weight='bold', family='Times New Roman')
    plt.yticks(fontsize=20, weight='bold', family='Times New Roman')

    x = np.linspace(0, 80, 20)
    y = x
    plt.plot(x, y, color='darkred', linewidth=2, linestyle='--', label='Bisect', alpha=1)

    plt.scatter(y_train_i, y_train_pred_i, color='dodgerblue', marker='>', label='Training', s=10, edgecolor='dodgerblue', alpha=0.9)
    plt.scatter(y_test_i, y_test_pred_i, color='red', marker='o', label='Testing', s=10, edgecolor='red', alpha=0.1)

    plt.text(0, 75, f'Train\nR${{^2}}$: {r2_train:.2f}\nRMSE: {rmse_train:.2f}\nMAE: {mae_train:.2f}', fontsize=20, weight='bold', color='blue', family='Times New Roman')
    plt.text(65, 0, f'Test\nR${{^2}}$: {r2_test:.2f}\nRMSE: {rmse_test:.2f}\nMAE: {mae_test:.2f}', fontsize=20, weight='bold', color='red', family='Times New Roman')
    plt.title(col_name, fontsize=22, weight='bold', family='Times New Roman')

    plt.legend(loc='upper center', frameon=False, prop={"weight": "bold", "family": "Times New Roman", "size": 14})
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.tight_layout()
    plt.show()

# Pearson correlation heatmap
combined_data = np.hstack((X, Y))
combined_df = pd.DataFrame(combined_data, columns=feature_names + target_labels)
dcorr = combined_df.corr(method='pearson')

plt.figure(figsize=(12, 8), dpi=500)
sns.heatmap(data=dcorr, linewidths=0.3, vmax=1, vmin=-1, annot=True, fmt=".3f",
            annot_kws={'size': 10, 'weight': 'bold'},
            cmap="coolwarm", cbar_kws={'label': 'Pearson Correlation Coefficients'})
plt.xticks(rotation=90, size=12)
plt.yticks(size=12)
plt.rc('font', family='Times New Roman', weight='bold', size=12)
plt.title("Feature and Target Pearson Correlation Heatmap", fontsize=16, weight='bold')
plt.show()

# SHAP analysis
start_time = time.time()
shap_values_list = []
models = []

for i in range(Y.shape[1]):
    model = RandomForestRegressor(n_estimators=67, max_depth=27, random_state=42)
    model.fit(X_train, y_train[:, i])
    models.append(model)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_list.append(shap_values)

for i, shap_values in enumerate(shap_values_list):
    plt.figure(figsize=(8, 5), dpi=600)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_size=(8, 5))
    plt.title(f'SHAP Summary Plot for {target_names[i]}', fontsize=22, fontname='Times New Roman')
    plt.xlabel('SHAP value', fontsize=22, fontname='Times New Roman')
    plt.ylabel('Features', fontsize=22, fontname='Times New Roman')
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=22)
    for label in cbar.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.tight_layout()
    plt.savefig(f'SHAP_Summary_{target_names[i]}.tiff', dpi=600, bbox_inches='tight')
    plt.show()

combined_shap_values = np.mean(np.array(shap_values_list), axis=0)
shap_df = pd.DataFrame(combined_shap_values, columns=feature_names)

plt.figure(figsize=(8, 5), dpi=600)
shap.summary_plot(shap_df.values, X_test, feature_names=feature_names, show=False)
plt.title('Combined SHAP Summary Plot', fontsize=22, fontname='Times New Roman')
plt.xlabel('SHAP value', fontsize=22, fontname='Times New Roman')
plt.ylabel('Features', fontsize=22, fontname='Times New Roman')
plt.xticks(fontsize=22, fontname='Times New Roman')
plt.yticks(fontsize=22, fontname='Times New Roman')
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=22)
for label in cbar.get_yticklabels():
    label.set_fontname('Times New Roman')
plt.show()
end_time = time.time()
print("Runtime:", end_time - start_time)

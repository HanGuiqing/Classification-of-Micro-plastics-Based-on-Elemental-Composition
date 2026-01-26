import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# Load validation data
validation_data = pd.read_excel('experimental validation.xlsx')
X_validation = validation_data.iloc[:, 0:6].values
y_validation = validation_data.iloc[:, 6:12].values

# Predict using trained model
y_validation_pred = model.predict(X_validation)

# Compute R² and RMSE
r2_validation = r2_score(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mean_squared_error(y_validation, y_validation_pred))

# Parity plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=700)
plt.xlabel('Actual value (%)', fontsize=20, weight='bold', family='Times New Roman')
plt.ylabel('Predicted value (%)', fontsize=20, weight='bold', family='Times New Roman')
plt.xticks(fontsize=20, weight='bold', family='Times New Roman')
plt.yticks(fontsize=20, weight='bold', family='Times New Roman')

x = np.linspace(0, 100, 20)
y = x
plt.plot(x, y, color='darkred', linewidth=2, linestyle='--', label='Bisect', alpha=1)

sns.set(color_codes=True)
plt.scatter(y_validation, y_validation_pred, color='red', marker='o', label='Testing', s=70, edgecolor='red', alpha=0.6)

plt.legend(loc='upper center', frameon=False, prop={"weight": "bold", "family": "Times New Roman", "size": 14})
plt.text(55, 0, f'Validation\nR${{^2}}$: {r2_validation:.2f}\nRMSE: {rmse_validation:.2f}', fontsize=20, weight='bold', color='dodgerblue', family='Times New Roman')
plt.text(75, 50, 'RF', fontsize=20, weight='bold', family='Times New Roman')

plt.rcParams['axes.facecolor'] = 'white'
ax = plt.gca()
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

plt.show()

import numpy as np
import pandas as pd


np.random.seed(42)

labels = ['PET', 'PP_PE', 'PVC', 'PS', 'PA', 'PC']

params = [
    {'mean': 33, 'std': 10},
    {'mean': 43, 'std': 15},
    {'mean': 6, 'std': 5},
    {'mean': 4, 'std': 5},
    {'mean': 2, 'std': 5},
    {'mean': 2, 'std': 5}
]


def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        nums = []
        for param in params:
            num = np.random.normal(loc=param['mean'], scale=param['std'])
            nums.append(max(0, num))
        nums = np.array(nums)
        nums /= nums.sum()
        nums *= 100
        data.append(np.round(nums, 1))
    return np.array(data)


data = generate_data()

df = pd.DataFrame(data, columns=labels)
df.to_csv('synthetic_environmental_distribution.csv', index=False)

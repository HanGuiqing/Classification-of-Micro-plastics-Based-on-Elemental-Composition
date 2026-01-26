import numpy as np
import pandas as pd
import itertools

np.random.seed(42)

labels = ['PET', 'PP_PE', 'PVC', 'PS', 'PA', 'PC', 'PBT', 'POM', 'PMMA', 'PLA']


def generate_combination_data(num_samples=1, num_numbers=1, target_sum=100):
    """
    Generate synthetic polymer composition data for a given number of components.

    This function enumerates all possible combinations of polymers with
    'num_numbers' components and assigns random mass fractions to the
    selected polymers. The mass fractions are normalized so that the
    total composition sums to 'target_sum'.

    Parameters
    ----------
    num_samples : int
        Number of random samples generated for each polymer combination.
    num_numbers : int
        Number of polymers included in each mixture (e.g., 1 for single
        polymer, 2 for binary mixtures, etc.).
    target_sum : float
        Total mass fraction (typically 100 for wt%).

    Returns
    -------
    pandas.DataFrame
        Synthetic composition dataset for the specified number of components.
    """

    data = []

    combinations = list(itertools.combinations(labels, num_numbers))

    for combo in combinations:
        for _ in range(num_samples):
            # Sample random positive values
            nums = np.random.uniform(0, 100, size=num_numbers)

            # Normalize values to satisfy the sum-to-one (compositional) constraint
            nums /= nums.sum()
            nums *= target_sum

            # Initialize all polymer fractions to zero
            entry = {label: 0 for label in labels}

            # Assign values to the selected polymers in the current combination
            for i, label in enumerate(combo):
                entry[label] = round(nums[i], 1)

            # Record the polymer combination for traceability
            entry['combination'] = ' + '.join(combo)

            data.append(entry)

    return pd.DataFrame(data)


data = generate_combination_data(num_samples=1, num_numbers=1)

data = data[[*data.columns[-1:], *data.columns[:-1]]]

data.to_csv('synthetic_single_component_compositions.csv', index=False)

print(data.head())

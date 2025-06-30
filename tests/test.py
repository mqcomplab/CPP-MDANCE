import random
import numpy as np
from mdance.tools import bts
import pandas as pd
def continuous_data():
    arr=np.array(
        [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4],
         [4.5, 0.6, 6.7, 7.8, 8.9, 0.4],
         [3.2, 4.3, 5.4, 6.5, 7.6, 8.7],
         [9.8, 0.9, 1.0, 2.1, 3.2, 4.3]]
    )
    return arr

def csv_to_numpy(filename="testData.csv"):
    """
    Reads a CSV file and converts it into a NumPy array.

    :param file_path: Path to the CSV file
    :return: NumPy array of the CSV data
    """
    try:
        df = pd.read_csv(filename, header=None)  # Read CSV using pandas
        np_array = df.to_numpy()  # Convert DataFrame to NumPy array
        return np_array
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

matrix = csv_to_numpy("small.csv")
cutoff=3
metric='MSD'
N_atoms=1
N=5

def diversity_selection(matrix, percentage: int, metric, N_atoms=1, start='medoid'):
    n_total = len(matrix)
    total_indices = np.array(range(n_total))
    
    if start =='medoid':
        seed = bts.calculate_medoid(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'outlier':
        seed = bts.calculate_outlier(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'random':
        seed = random.randint(0, n_total - 1)
        selected_n = [seed]
    elif isinstance(start, list):
        selected_n = start
    else:
        raise ValueError('Select a correct starting point: medoid, outlier, \
                         random or outlier')

    n = len(selected_n)
    n_max = int(np.floor(n_total * percentage / 100))
    if n_max > n_total:
        raise ValueError('Percentage is too high')
    selection = [matrix[i] for i in selected_n] 
    selection = np.array(selection)
    selected_condensed = np.sum(selection, axis=0)
    if metric == 'MSD':
        sq_selection = selection ** 2
        sq_selected_condensed = np.sum(sq_selection, axis=0)
        print(sq_selected_condensed)
    
    while len(selected_n) < n_max:
        select_from_n = np.delete(total_indices, selected_n)
        print(selected_condensed)
        print(sq_selected_condensed)
        print(select_from_n)
        print("-------")
        if metric == 'MSD':
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed, sq_selected_condensed=sq_selected_condensed, n=n, select_from_n=select_from_n, N_atoms=N_atoms)
            sq_selected_condensed += matrix[new_index_n] ** 2
        else:
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed, n=n, select_from_n=select_from_n)
        selected_condensed += matrix[new_index_n]
        selected_n.append(new_index_n)
        n = len(selected_n)
        print("++++++++++")
    return selected_n


def get_new_index_n(matrix, metric, selected_condensed, n, select_from_n, **kwargs):
    if 'sq_selected_condensed' in kwargs:
        sq_selected_condensed = kwargs['sq_selected_condensed']
    if 'N_atoms' in kwargs:
        N_atoms = kwargs['N_atoms']
    
    # Number of fingerprints already selected and the new one to add
    n_total = n + 1
    
    # Placeholders values
    min_value = -1
    index = len(matrix) + 1
    
    # Calculate MSD for each unselected object and select the index with the highest value.
    for i in select_from_n:
        if metric == 'MSD':
            sim_index = bts.extended_comparison([selected_condensed + matrix[i], sq_selected_condensed + (matrix[i] ** 2)], data_type='condensed', metric=metric, N=n_total, N_atoms=N_atoms) 
            print(sim_index)
        else:
            sim_index = bts.extended_comparison([selected_condensed + matrix[i]], data_type='condensed', metric=metric, N=n_total)
        if sim_index > min_value:
            min_value = sim_index
            index = i
        else:
            pass
    return index

print(diversity_selection(matrix, 40, metric, N_atoms))
print(bts.diversity_selection(matrix, 40, metric, N_atoms))

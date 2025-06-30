import numpy as np
import pandas as pd
import csv
import sys
from mdance.tools import bts
from mdance import data

def bit_data():
    arr = np.array(
        [[0, 1, 0, 0, 1, 0],
         [1, 0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1, 1],
         [0, 1, 1, 0, 1, 1]]
    )
    numpy_to_csv(arr, "bit.csv")
    return arr

def continuous_data():
    arr=np.array(
        [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
         [7.8, 8.9, 9.0, 1.2, 2.3, 3.4],
         [4.5, 0.6, 6.7, 7.8, 8.9, 0.4],
         [3.2, 4.3, 5.4, 6.5, 7.6, 8.7],
         [9.8, 0.9, 1.0, 2.1, 3.2, 4.3]]
    )
    numpy_to_csv(arr, "continuous.csv")
    return arr

def sim_data():
    arr = np.load(data.sim_traj_numpy)
    numpy_to_csv(arr, "sim.csv")
    return arr

def generate_matrix(rows, cols, filename="testData.csv"):
    """Generate a matrix of random floats and save to CSV."""
    matrix = np.random.rand(rows, cols).astype(np.float32)  # Generate random float32 values
    numpy_to_csv(matrix, filename)
    return matrix


def numpy_to_csv(matrix, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
#    print(f"Matrix saved to {filename}")

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

def str_parse(val):
    if round(val) - val > -0.00005 and round(val) - val < 0.00005:
        return str(round(val))
    return str.format('{0:.6}', val)

def print_matrix(mat):
    out = "[ "
    for i in range(len(mat)):
        for j in range(len(mat[i])-1):
            out+= str_parse(mat[i][j]) + ", "
        if i == len(mat)-1:
            out += str_parse(mat[i][-1]) + " ]"
        else:
            out += str_parse(mat[i][-1]) + "; "
    return out

def print_vector(vec):
    out = "[ "
    for i in range(len(vec)-1):
        out+= str_parse(vec[i]) + ", "
    out += str_parse(vec[-1]) + " ]"
    return out

def run_tests(matrix, N_atoms):
    msd = bts.mean_sq_dev(matrix, N_atoms=N_atoms)
    ec = bts.extended_comparison(matrix, 'full', 'MSD', N_atoms=N_atoms)
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(matrix**2, axis=0)
    ecc = bts.extended_comparison((c_sum, sq_sum), 'condensed', 'MSD', len(matrix), N_atoms=N_atoms)
    compSim = bts.calculate_comp_sim(matrix, 'MSD', N_atoms=N_atoms)
    medoid = bts.calculate_medoid(matrix, 'MSD', N_atoms)
    outlier = bts.calculate_outlier(matrix, 'MSD', N_atoms)
    trim = bts.trim_outliers(matrix, 0.6, 'MSD', N_atoms)
    diversitySel = bts.diversity_selection(matrix, 40, 'MSD', N_atoms=N_atoms)
    distances = bts.refine_dis_matrix(matrix)
    print(str.format('{0:.6}', msd))
    print(str.format('{0:.6}', ec))
    print(str.format('{0:.6}', ecc))
    print(medoid)
    print(outlier)
    print(print_vector(compSim))
    print(len(trim))
    print(print_matrix(trim))
    print(diversitySel)
    print(distances)

# Run tests
matrix = bit_data()
run_tests(matrix, 1)
matrix = continuous_data()
run_tests(matrix, 2)
#matrix = sim_data()
#run_tests(matrix, 50)
matrix = generate_matrix(10, 20, "small.csv")
run_tests(matrix, 3)
#matrix = csv_to_numpy("1d.csv")
#run_tests(matrix, 1)
# data = generate_matrix(500,20, "medium.csv")
# data = generate_matrix(2000,2000, "big.csv")


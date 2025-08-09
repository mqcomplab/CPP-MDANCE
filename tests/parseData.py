import numpy as np
import pandas as pd
import csv
import sys
import time
from mdance.tools import bts
from mdance import data

def generate_matrix(rows, cols, filename="testData.csv"):
    """Generate a matrix of random floats and save to CSV."""
    matrix = np.random.rand(rows, cols).astype(np.float32)  # Generate random float32 values
    numpy_to_csv(matrix, filename)
    return matrix


def numpy_to_csv(matrix, filename):
    with open("data/"+filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
    print("\n" + filename + "\n----------------------")
    print("\n" + filename + "\n----------------------", file=sys.stderr)

def csv_to_numpy(filename="testData.csv"):
    """
    Reads a CSV file and converts it into a NumPy array.

    :param file_path: Path to the CSV file
    :return: NumPy array of the CSV data
    """
    try:
        df = pd.read_csv("data/"+filename, header=None)  # Read CSV using pandas
        np_array = df.to_numpy()  # Convert DataFrame to NumPy array
        print("\n" + filename + "\n----------------------")
        print("\n" + filename + "\n----------------------", file=sys.stderr)
        return np_array
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def str_parse(val):
    # if round(val) - val > -0.00005 and round(val) - val < 0.00005:
    #     return str(round(val))
    return str.format('{0:.10g}', val)

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
    # MSD test
    t_start = time.perf_counter()
    msd = bts.mean_sq_dev(matrix, N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("msd: " + str.format('{0:.10g}', float(msd)))
    print("msd: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # EC test
    t_start = time.perf_counter()
    ec = bts.extended_comparison(matrix, 'full', 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("ec: " + str.format('{0:.10g}', float(ec)))
    print("ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # condensed EC test
    N = len(matrix)
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(matrix ** 2, axis=0)
    t_start = time.perf_counter()
    ec = bts.extended_comparison((c_sum, sq_sum), 'condensed', 'MSD', N, N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("condensed ec: " + str.format('{0:.10g}', float(ec)))
    print("condensed ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # Esim EC test
    t_start = time.perf_counter()
    ec = bts.extended_comparison([c_sum], 'condensed', metric='RR', N=N, c_threshold=None, w_factor='fraction')
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("esim ec: " + str.format('{0:.10g}', float(ec)))
    print("esim ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)
    
    # calcMedoid test
    t_start = time.perf_counter()
    idx = bts.calculate_medoid(matrix, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("calcMedoid: " + str(idx))
    print("calcMedoid: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # calcOutlier test
    t_start = time.perf_counter()
    idx = bts.calculate_outlier(matrix, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("calcOutlier: " + str(idx))
    print("calcOutlier: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # trimOutlier test
    t_start = time.perf_counter()
    var = bts.trim_outliers(matrix, 0.1, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("trimOutlier: " + print_matrix(var))
    print("trimOutlier: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # compSim tests
    mts = {0: 'MSD', 1: 'BUB', 2: 'Fai', 3: 'Gle', 4: 'Ja', 5: 'JT', 6: 'RT', 7: 'RR', 8: 'SM', 9: 'SS1', 10: 'SS2'}
    for mt in range(11):
        t_start = time.perf_counter()
        val = bts.calculate_comp_sim(matrix, mts[mt], N_atoms=N_atoms)
        t_end = time.perf_counter()
        t_dif = t_end-t_start
        print(mts[mt], "compSim: " + print_vector(val))
        print(mts[mt], "compSim: " + str.format('{0:.10g}', t_dif), file=sys.stderr)


# Run tests
matrix = csv_to_numpy("bit.csv")
run_tests(matrix, 1)
matrix = csv_to_numpy("continuous.csv")
run_tests(matrix, 2)
matrix = csv_to_numpy("sim.csv")
run_tests(matrix, 50)
matrix = generate_matrix(10, 20, "small.csv")
# matrix = csv_to_numpy("small.csv")
run_tests(matrix, 3)
# matrix = csv_to_numpy("1d.csv")
# run_tests(matrix, 1)
matrix = generate_matrix(200, 500, "mid.csv")
# matrix = csv_to_numpy("mid.csv")
run_tests(matrix, 3)
matrix = generate_matrix(50000, 500, "large.csv")
# matrix = csv_to_numpy("large.csv")
run_tests(matrix, 3)


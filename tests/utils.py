import numpy as np
from prettytable import PrettyTable
import os

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    times = []
    costs = []

    for line in lines:
        _, t, _, c = line.strip().split()

        times.append(float(t))
        costs.append(float(c))

    return np.mean(times), np.std(times), np.mean(costs), np.std(costs)

def read_folder(folder_path):
    configs = os.listdir(folder_path)
    results_dict = {}
    for config in configs:
        m, n, _ = config.split("_")
        path = os.path.join(folder_path, config)
        mcc, tsp = sorted(os.listdir(path))
        mcc = read_file(os.path.join(path, mcc))
        tsp = read_file(os.path.join(path, tsp))
        results_dict[config] = [f"Grid ({m}, {n})", list(mcc), list(tsp)]

    return results_dict

def analyse_results(fpath):

    print('\033[1m' + "Analysing results from `{}`".format(fpath) + '\033[0m')
    results_dict = read_folder(fpath)

    mcc_table = PrettyTable()
    mcc_table.title = "MCC Results"
    mcc_table.field_names = ["Config", "T_mean (s)", "T_std (s)", "C_mean (m)", "C_std (m)"]
    
    tsp_table = PrettyTable()
    tsp_table.title = "TSP Results"
    tsp_table.field_names = ["Config", "T_mean (s)", "T_std (s)", "C_mean (m)", "C_std (m)"]

    ratio_table = PrettyTable()
    ratio_table.title = "Relative Performance Statistics"
    ratio_table.field_names = ["Config", "T (%)", "C (%)"]


    for val in results_dict.values():
        config, mcc, tsp = val
        mcc_table.add_row([config] + [round(i, 4) for i in mcc])
        tsp_table.add_row([config] + [round(i, 4) for i in tsp])

        tp = round((mcc[0] - tsp[0])/(tsp[0]) * 100, 4)
        cp = round((mcc[2] - tsp[2])/(tsp[2]) * 100, 4)

        ratio_table.add_row([config, tp, cp])
    
    print(mcc_table)
    print()
    print(tsp_table)
    print()
    print(ratio_table)

    # Create a file `analysis.txt` as print eveything there
    file_path = os.path.join(fpath, 'analysis.txt')
    with open(file_path, 'w') as f:
        f.write(str(mcc_table))
        f.write("\n")
        f.write(str(tsp_table))
        f.write("\n")
        f.write(str(ratio_table))
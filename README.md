# Multiple Cluster Coverage
This repository contains the code for the __Multiple Cluster Coverage__ algorithm developed as a part of my _Bachelor's thesis_ at __RBCCPS, IISc, Bangalore__.

## Installation Instructions
1. Clone the repository to your local machine.
2. Use pip to install the repository
```bash
cd mcc/
python -m pip install -e .
```
This installs all the required modules required for the repository.

## Benchmark
1. Run `tests/benchmark.py` to run the benchmark script. The script compares the performance of our algorithm with the TSP solver from Google OR Tools.
```bash
cd mcc
python tests/benchmark.py /path/to/config.yaml
```

2. The script requires the path to be a configuration file. A sample configuration file can be found [here](./testcase.yaml). The output of the benchmark script ran using this configuration file can be found in the [results](./results/) directory.

## Results
```
+---------------------------------------------------------------+
|                          MCC Results                          |
+-------------+------------+-----------+------------+-----------+
|    Config   | T_mean (s) | T_std (s) | C_mean (m) | C_std (m) |
+-------------+------------+-----------+------------+-----------+
| Grid (1, 2) |   0.0042   |   0.0002  |  36.2886   |   0.189   |
| Grid (2, 2) |   0.0136   |   0.0003  |  94.6935   |   2.445   |
| Grid (3, 5) |   0.1857   |   0.005   |  335.3863  |   1.744   |
+-------------+------------+-----------+------------+-----------+
+---------------------------------------------------------------+
|                          TSP Results                          |
+-------------+------------+-----------+------------+-----------+
|    Config   | T_mean (s) | T_std (s) | C_mean (m) | C_std (m) |
+-------------+------------+-----------+------------+-----------+
| Grid (1, 2) |   0.0239   |   0.0013  |  35.9227   |   0.0345  |
| Grid (2, 2) |   0.1326   |   0.0202  |   82.171   |   0.0634  |
| Grid (3, 5) |   3.022    |   0.573   |  297.5467  |   3.893   |
+-------------+------------+-----------+------------+-----------+
+----------------------------------+
| Relative Performance Statistics  |
+-------------+----------+---------+
|    Config   |  T (%)   |  C (%)  |
+-------------+----------+---------+
| Grid (1, 2) | -82.4804 |  1.0187 |
| Grid (2, 2) | -89.7276 | 15.2395 |
| Grid (3, 5) | -93.8561 | 12.7172 | 
+-------------+----------+---------+
```
__Note__: Here (-) sign indicates the better performance of the algorithm compared to the benchmark
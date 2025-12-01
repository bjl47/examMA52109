This is a read me file outlining all changes done to this file, according to tasks:


Task 1:
Corrections in file algorithm.py
- LINE 26: fixed to k > n_samples
- LINE 30: fixed size of k to be k = k (iterating over itself, not k+1)
- LINE 42: changed kwarg axis=1 to axis = 2
- LINE43: fixed argmax to argmin for kmeans algorithm

Task 2: 
- Corrections done in demo/cluster_plot.py and cluster_maker/interface.py
- Mainly logical errors for k, where functions were misassigning the wrong values to k
- Explanations done in EXPLANATION.md, and output saved to demo_output

Task 3:
- new test_preprocessing file creates in tests, with 3 tests written to assess the functions in preprocessing.py

Task 4:
- New script simulated_clustering.py created in demo that uses cluster_maker package to analyse and visualise the simulated_data.csv, saved in demo_output

Task 5:
- New agglomerative.py script is written in cluster_maker, which craetes a new agglomerative function using AgglomerativeClustering package
- Output saved to demo_output
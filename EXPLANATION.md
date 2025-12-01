# INSTRUCTIONS FOR TASK 2
# 2) The script `cluster_plot.py` in folder `demo/` runs without crashing
## with the following command line:
##      python demo/cluster_plot.py data/demo_data.csv
## The script, though, does not do what it was designed to do. For this task
## you must:
##  a. Read and run the demo script.
##  b. Work out what the script is currently doing, and what it was 
##     intended to do.
##  c. Identify and correct the mistake in the script so that it behaves 
##     as intended. Re-run it to check that it now works correctly.
##  d. Create a file called `EXPLANATION.md` in the project root. In this file:
##      - briefly explain what was wrong with the original script and how you 
##        fixed it;
##      - summarise what the corrected demo script now does;
##      - give a short overview of what the `cluster_maker` package does, 
##        describing the purpose of its main components (e.g. preprocessing, 
##        algorithms, plotting, interface).


This is what the cluster_plot demo script does/inteded to do:

1. Run clustering for k = 2, 3, 4, 5 using run_clustering from the cluster_maker's interface package.
2. Save clustered CSVs for each k.
3. Save cluster plots for each k.
4. Collect metrics (inertia, silhouette) for each k and save as CSV.
5. Plot silhouette score vs k.

- However, the issue is that in line 62, the k was capped at 3 because k was assigned k = min(k, 3). So, we only see 3 clusters maximum in the demo_output files.
- Therefore, to fix this, we assign k = k so that k will iterate and we should see more clusters in the demo_output plots.
- Furthermore, the interface.py script had an error in it -- line 25 put the default of k as 3, when it should be 2 (minimum number of clusters)
- After fixing both scripts, we now see the correct number of clusters in the output plot.


This is what the cluster_maker package does:
1. Preprocessing: 
- The preprocessing.py functions select a subset of numeric columns to use as features and standardises features to zero mean and unit variance.

2. Main clustering functions:
- the algorithms package implements the main clustering logic, including centroid initalisation, assigning points to clusters, calculates k-means, and then updating the centroids from the MSE of the points in the cluster.

3. Analysis:
- The evaluation.py has functions that computes the sum of squared distances (intertia) within the clusters, measures silheoutte score, and determines optimal number of clusters with the elbow curve function.

4. Plotting and exporting:
- Plotting_clustered.py visualises the clustering results 
- Data_exporter.py saves the clustering results, including silehoutte score and inertia to csvs

5. Interface:
- Lastly, the interface wraps preprocessing, clustering, evaluation, plotting and export into one function for efficient execution
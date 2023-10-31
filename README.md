# RFVM
At this repository we present the model **Relevance Feature and Vector Machine (RFVM)** along with an explanatory example with a toy-problem to understand how to launch and use the algorithm. The explanatory notebook is located at RFVM_example.ipynb and the algorithm source code is located at RFVM_prune_elbo_dep.py.

The main goal of this notebook is to teach ML users how to use the algorithm presented at (FILL WITH PAPER WHEN PUBLISHED).

Also, this model is designed to work within fat-data scenarios, that is, databases with much more features than samples (D>>N). It includes both a feature and relevance vector selection procedures, i.e., it reduces the database to a compact solution with the minimum number of features to distinguish between the classes and the minimum number of samples to describe the sample distribution. However, it present a main limitation due to its formulation: Data with much more samples than features. Hence, within this scenario, if we want to achieve a compact solution on features and relevance vector, we recomended other approaches such as Sparse Gaussian Processes with ARD or Support Vector Machines with L1 regulariation.

Enjoy!

# config.py

SEED = 42  # Ensuring reproducibility

#fcl PROBLEM 
DATA_FILE = "p1"


NUM_REPLICATIONS = 10 

#Making FLC problem stochastic by defining scenarios for demand 
NUM_SCENARIOS = 60 #Number of demand scenario generated
VARIANCE_FACTOR=0.2
P_MEDIAN = 5 #Number of sampled subproblems 

#Ml-augmented Benders structure
USE_PREDICTION = True # If true, use ML for creation of cuts on unsampled subproblems
SOLVE_UNSELECTED_IF_NO_CUTS = True  # If true, it will solve unsampled subproblems if other conditions like no ml generated cuts are met. 
PREDICTION_METHOD = "knn" # ML type uses for cut generation of unsampled subproblems
N_NEIGHBORS = 3 #How many nearest neighbor to consider if using KNN method. 

#Solving scenario for getting input for feature generation using random walk
N_F = 40  # Number of y_values_features
BINARY = False #Generation of y values to solve different subproblems for the final purpose of feature creation
#the forer was 20 


#Random Walk 
N_WALK = 10 #Number of random walks to geenrate from each node
L_WALK = 20 # Length of the random walk 
W = 5

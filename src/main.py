import os
import numpy as np
import random
from data import read_dataset
from master_problem import solve_stochastic_cflp
from sub_problem import solve_subproblem
from sampled_subproblems import solve_p_median
from learn_features import learn_subproblem_features
from random_walk_generation import generate_random_walks
from itertools import product
import pandas as pd



# Set a global seed for reproducibility
SEED = 42  # You can change this seed to experiment with different randomness
np.random.seed(SEED)
random.seed(SEED)

def generate_y_values(n_f, data, binary=False, seed=42):
    """
    Generate n_f sets of |J| values for y_j that satisfy the constraint (feasibility constraint):
        sum(data.capacities[j] * y_j for j in data.J) >= data.max_demand_sum_over_scenario

    Parameters:
        n_f (int): Number of sets to generate.
        data: An object containing J, capacities, and max_demand_sum_over_scenario.
        binary (bool): If True, values are binary {0,1}. If False, values are continuous in [0,1].
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: A (n_f, |J|) matrix with generated values.
    """
    np.random.seed(seed)  # Ensuring consistent random values
    y_values = np.zeros((n_f, len(data.J)))  # Initialize matrix

    for i in range(n_f):
        valid = False
        while not valid:
            if binary:
                y_row = np.random.choice([0, 1], size=len(data.J))
            else:
                y_row = np.random.uniform(0, 1, size=len(data.J))

            # Check constraint
            if sum(data.capacities[j] * y_row[j] for j in data.J) >= data.max_demand_sum_over_scenario:
                valid = True
                y_values[i] = y_row  # Store the valid row

    return y_values


def calculate_dual(y_values, data):
    """
    Calculate and store G_s(x_i) for all s in S.

    Parameters:
        y_values (np.ndarray): The set of first-stage decision values.
        data: Problem data containing set S.

    Returns:
        dict: A dictionary storing (mu, nu) for each (y, s).
    """
    first_stage_decision_count = y_values.shape[0]  # Get the number of y decision vectors

    # Dictionary to store results
    dual_values = {}

    for count in range(first_stage_decision_count):  
        for s in data.S:  
            obj, mu, nu = solve_subproblem(data, y_values[count], s)  
            dual_values[(count, s)] = (mu, nu)  

    return dual_values  # Return stored results


def distance_function(mu_nu_si, mu_nu_sj):
    """
    Compute the normalized Euclidean distance between two pairs of (mu, nu).

    Parameters:
        mu_nu_si (tuple): (mu, nu) for scenario s_i.
        mu_nu_sj (tuple): (mu, nu) for scenario s_j.

    Returns:
        float: The computed Euclidean distance, normalized in [0,1].
    """
    mu_si, nu_si = mu_nu_si
    mu_sj, nu_sj = mu_nu_sj

    keys_mu = sorted(mu_si.keys())
    keys_nu = sorted(nu_si.keys())

    mu_si_values = np.array([mu_si[key] for key in keys_mu])
    mu_sj_values = np.array([mu_sj[key] for key in keys_mu])

    nu_si_values = np.array([nu_si[key] for key in keys_nu])
    nu_sj_values = np.array([nu_sj[key] for key in keys_nu])

    vec_si = np.concatenate([mu_si_values, nu_si_values])
    vec_sj = np.concatenate([mu_sj_values, nu_sj_values])

    distance = np.linalg.norm(vec_si - vec_sj)
    norm_factor = np.linalg.norm(vec_si) + np.linalg.norm(vec_sj)

    normalized_distance = distance / norm_factor if norm_factor > 0 else 0
    return normalized_distance


def calculate_weight(dual_values, y_values):
    """
    Calculate weight(s_i, s_j) for each pair of different scenarios s_i, s_j.

    Parameters:
        dual_values (dict): A dictionary storing (mu, nu) for each (count, s).
        y_values (np.ndarray): First-stage decision values.

    Returns:
        dict: A dictionary storing weights for each pair (s_i, s_j).
    """
    first_stage_decision_count = y_values.shape[0]
    scenarios = set(s for _, s in dual_values.keys())  
    weight_values = {}  

    for s_i in scenarios:
        for s_j in scenarios:
            if s_i != s_j:
                total_difference = 0  

                for count in range(1, first_stage_decision_count + 1):
                    mu_nu_si = dual_values.get((count, s_i))
                    mu_nu_sj = dual_values.get((count, s_j))

                    if mu_nu_si is not None and mu_nu_sj is not None:
                        total_difference += distance_function(mu_nu_si, mu_nu_sj)

                weight_values[(s_i, s_j)] = total_difference / first_stage_decision_count

    return weight_values


# Set a global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Define parameter grid
data_files = ["p1"]  # List of dataset file names
use_prediction_options = [True, False]
variance_factors = [0.5,1.5]
p_median_values = [5,10]
NUM_SCENARIOS=60
N_F = 20

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "../data/")
output_file = os.path.join(script_dir, "stochastic_cflp_results.xlsx")

# DataFrame to store results
results = []

for DATA_FILE, USE_PREDICTION, VARIANCE_FACTOR, P_MEDIAN_VALUE in product(
    data_files, use_prediction_options, variance_factors, p_median_values
):
    
    datafile = os.path.join(DATA_DIR, DATA_FILE)
    data = read_dataset(datafile, num_scenarios=NUM_SCENARIOS, variance_factor=VARIANCE_FACTOR)

    # Generate y values
    y_values_features = generate_y_values(n_f=N_F, data=data, binary=False, seed=SEED)
    Dual_values_features = calculate_dual(y_values_features, data)
    weight_values_features = calculate_weight(Dual_values_features, y_values_features)

    # Generate scenario features
    C = generate_random_walks(weight_values_features, n_walk=10, l_walk=20, seed=SEED)
    model = learn_subproblem_features(C, w=5)
    feature_vectors = {scenario: model.wv[scenario] for scenario in model.wv.index_to_key}

    # Solve p-median problem
    selected_scenarios, assignments = solve_p_median(feature_vectors, p=P_MEDIAN_VALUE)

    # Solve the stochastic CFLP
    solution = solve_stochastic_cflp(
        selected_scenarios, feature_vectors, data,
        prediction_method="knn", n_neighbors=5, use_prediction=USE_PREDICTION
    )

    # Store results
    results.append({
        "DATA_FILE": DATA_FILE,
        "USE_PREDICTION": USE_PREDICTION,
        "VARIANCE_FACTOR": VARIANCE_FACTOR,
        "P_MEDIAN_VALUE": P_MEDIAN_VALUE,
        "Objective Value": solution.objective_value,
        "Open Facilities": [j for j in data.J if solution.locations[j] > 0.5],
        "Solution Time (sec)": solution.solution_time,
        "BD Cuts (MIP) Selected": sum(solution.num_cuts_mip_selected.values()) if solution.num_cuts_mip_selected else 0,
        "BD Cuts (Relaxed) Selected": sum(solution.num_cuts_rel_selected.values()) if solution.num_cuts_rel_selected else 0,
        "BD Cuts (MIP) ML": sum(solution.num_cuts_mip_ml.values()) if solution.num_cuts_mip_ml else 0,
        "BD Cuts (Relaxed) ML": sum(solution.num_cuts_rel_ml.values()) if solution.num_cuts_rel_ml else 0,
        "BD Cuts (MIP) Unselected": sum(solution.num_cuts_mip_unselected.values()) if solution.num_cuts_mip_unselected else 0,
        "BD Cuts (Relaxed) Unselected": sum(solution.num_cuts_rel_unselected.values()) if solution.num_cuts_rel_unselected else 0,
        "Explored Branch-and-Bound Nodes": solution.num_bnb_nodes,
    })

# Convert results to a DataFrame and save to Excel
df_results = pd.DataFrame(results)
df_results.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")

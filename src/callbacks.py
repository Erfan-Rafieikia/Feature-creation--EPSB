from data import Data
from gurobipy import GRB, quicksum
from sub_problem import solve_subproblem
from dual_train import train_value  # Import training function
import numpy as np


class Callback:
    def __init__(self, dat: Data, y, eta, selected_scenarios, feature_vectors, prediction_method="regression", n_neighbors=5, use_prediction=True):
        self.dat = dat
        self.y = y
        self.eta = eta
        self.selected_scenarios = set(selected_scenarios)  
        self.feature_vectors = feature_vectors  
        self.model_mu = None  
        self.model_nu = None  
        self.num_cuts_mip_selected = {s: 0 for s in range(len(dat.S))}
        self.num_cuts_rel_selected = {s: 0 for s in range(len(dat.S))}
        self.num_cuts_mip_ml = {s: 0 for s in range(len(dat.S))}
        self.num_cuts_rel_ml = {s: 0 for s in range(len(dat.S))}
        self.num_cuts_mip_unselected = {s: 0 for s in range(len(dat.S))}
        self.num_cuts_rel_unselected = {s: 0 for s in range(len(dat.S))}
        self.prediction_method = prediction_method
        self.n_neighbors = n_neighbors
        self.use_prediction = use_prediction  

    def __call__(self, mod, where):
        if where == GRB.Callback.MIPSOL:
            y_values = mod.cbGetSolution(self.y)  
            eta_value = mod.cbGetSolution(self.eta)  
            solved_duals = {}
            cuts_added_selected = False  
            cuts_added_ml = False
            cuts_added_unselected = False

            # Solve subproblems for selected scenarios
            for s in self.selected_scenarios:
                obj, mu, nu = solve_subproblem(self.dat, y_values, s)
                solved_duals[s] = (mu, nu)
                if obj > eta_value[s]:
                    self.add_optimality_cut(mod, mu, nu, s)
                    self.num_cuts_mip_selected[s] += 1
                    cuts_added_selected = True
            
            if self.use_prediction:
                self.model_mu, self.model_nu = train_value(
                    solved_duals, self.feature_vectors, 
                    method=self.prediction_method, n_neighbors=self.n_neighbors
                )
                unselected_scenarios = set(self.dat.S) - self.selected_scenarios
                predicted_duals = self.predict_duals(unselected_scenarios)

                for s in unselected_scenarios:
                    mu, nu = predicted_duals[s]
                    obj_pred = self.compute_predicted_obj(mu, nu, s, mod)  
                    eta_s_val = mod.cbGetSolution(self.eta[s])
                    if obj_pred > eta_s_val:
                        self.add_optimality_cut(mod, mu, nu, s)
                        self.num_cuts_mip_ml[s] += 1
                        cuts_added_ml = True
            else:
                # If use_prediction is False, solve all unselected scenarios like selected
                unselected_scenarios = set(self.dat.S) - self.selected_scenarios
                for s in unselected_scenarios:
                    obj, mu, nu = solve_subproblem(self.dat, y_values, s)
                    if obj > eta_value[s]:
                        self.add_optimality_cut(mod, mu, nu, s)
                        self.num_cuts_mip_unselected[s] += 1
                        cuts_added_unselected = True
            
            # If use_prediction is True and no cuts were added from selected and ML, solve unselected scenarios like selected
            if self.use_prediction and not cuts_added_selected and not cuts_added_ml:
                for s in unselected_scenarios:
                    obj, mu, nu = solve_subproblem(self.dat, y_values, s)
                    if obj > eta_value[s]:
                        self.add_optimality_cut(mod, mu, nu, s)
                        self.num_cuts_mip_unselected[s] += 1
                        cuts_added_unselected = True
                        break  # Immediately return to the master problem

    def predict_duals(self, unselected_scenarios):
        if self.model_mu is None or self.model_nu is None:
            raise ValueError("Prediction models for mu and nu are not trained.")
        predicted_duals = {}
        for s in unselected_scenarios:
            X_test = np.array([self.feature_vectors[s]])
            mu_pred = self.model_mu.predict(X_test).flatten()
            nu_pred = self.model_nu.predict(X_test).flatten()
            predicted_duals[s] = (mu_pred, nu_pred)
        return predicted_duals

    def compute_predicted_obj(self, mu, nu, scenario, mod):
        obj = sum(float(self.dat.scenarios[scenario, i]) * float(mu[i]) for i in self.dat.I)
        obj += sum(
            float(self.dat.capacities[j]) * float(nu[j]) * float(mod.cbGetSolution(self.y[j]))
            for j in self.dat.J
        )
        return obj

    def add_optimality_cut(self, mod, mu, nu, scenario):
        rhs = quicksum(self.dat.scenarios[scenario, i] * mu[i] for i in self.dat.I)
        rhs += quicksum(self.dat.capacities[j] * nu[j] * self.y[j] for j in self.dat.J)
        mod.cbLazy(self.eta[scenario] >= rhs)

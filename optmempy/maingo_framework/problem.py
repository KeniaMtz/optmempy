import maingopy
from pathlib import Path
import joblib
import numpy as np

from optmempy.maingo_framework.framework import (
  ANNWrapper,
  get_bounds,
  separation_factor_objective,
  molar_power_objective,
  divalent_recovery_constraint,
  separation_factor_constraint,
  pressure_constraint,
)
from optmempy.maingo_framework.utils import (
  scale_x,
  get_ann_input_vector,
  get_idx_map,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "nf_cascade_ann"

class NF_MAiNGO_Model(maingopy.MAiNGOmodel):
    def __init__(self, pm, pp, objective_type="separation_factor", constraint_value=0.6):
        super().__init__()
        
        opt_problems = ['separation_factor', 'molar_power']
        if objective_type not in opt_problems:
            raise ValueError(f'objective_type must be one of {opt_problems}')
        
        self.objective_type = objective_type
        self.constraint_value = constraint_value

        scaler_dir = DATA_DIR / "data"
        self.x_scaler = joblib.load(scaler_dir / "input_scaler.joblib")
        self.y_scaler = joblib.load(scaler_dir / "output_scaler.joblib")
        data = np.load(scaler_dir / "scaled_data.npz")
        self.scale_idx = data["scale_idx"]

        c_y = np.zeros((1, 3))
        if self.objective_type == "separation_factor":
            c_y[0, 0] = constraint_value  # recovery
            self.R_min_scaled = self.y_scaler.transform(c_y)[0, 0]
        elif self.objective_type == "molar_power":
            c_y[0, 1] = constraint_value  # separation factor
            self.Beta_min_scaled = self.y_scaler.transform(c_y)[0, 1]

        self.pm = pm
        self.pp = pp
        
        self.lb, self.ub = get_bounds(pm, self.x_scaler, self.scale_idx) # Bounds in the scaled space
        self.n_var = len(self.lb)

        self.ann_model = ANNWrapper(
            ann_path=DATA_DIR,
            ann_name="fixed_NF_config"
        )

        self.x_vector_real = get_ann_input_vector(self.pm, self.pp)
        self.x_vector_scaled = list(scale_x(self.x_vector_real, self.x_scaler, self.scale_idx))
        

    def get_variables(self):
        return [
            maingopy.OptimizationVariable(
                maingopy.Bounds(self.lb[i], self.ub[i]),
                maingopy.VT_CONTINUOUS,
                f"x_{i}"
            ) for i in range(self.n_var)
        ]

    def evaluate(self, vars):
        result = maingopy.EvaluationContainer()
        vars_list = list(vars)
        
        ns = self.pm["nst"]

        start = 0
        p_feed = vars_list[start]
        start += 1
        pp_list = vars_list[start : start + ns]

        idx = get_idx_map(ns)
        x_scaled = self.x_vector_scaled.copy()
        pf0, _ = idx["p_feed"]
        pp0, pp_1 = idx['pp']
        x_scaled[pf0] = p_feed
        x_scaled[pp0:pp_1] = pp_list

        # Objective
        if self.objective_type == "separation_factor":
            result.objective = separation_factor_objective(x_scaled, self.ann_model)
        elif self.objective_type == "molar_power":
            result.objective = molar_power_objective(x_scaled, self.ann_model)
        
        # Constraints
        if self.objective_type == "separation_factor":
            c_main = divalent_recovery_constraint(x_scaled, self.ann_model, self.R_min_scaled)
        elif self.objective_type == "molar_power":
            c_main = separation_factor_constraint(x_scaled, self.ann_model, self.Beta_min_scaled)
        c_pres = pressure_constraint(x_scaled, self.pm)
        
        result.ineq = [c_main] + c_pres

        return result
import pandas as pd
import numpy as np
from optmempy.ipopt_framework import opti_initialize as initialize
from optmempy.ipopt_framework import opti_model as mdl
from optmempy.ipopt_framework.utils_ipopt import pyomo_to_np_array

PERMEATE_PRESSURES = []
LAMBDA = []
PI = []
DILUTION_FRACTIONS = []
FEED_FRACTIONS = []
LAMBDA_NORM = []
PI_NORM = []
pm, pm_sim = initialize.load_problem(n_stages=6,
                               transport_model="sdec",
                               relax=True,
                               dp_max=22.4,
                               model_x=True,
                               T = 299.65)
process_parameters = {
  'p_feed'          :  22.4e5,  # Pa
  'pp_list'  : [0, 0, 0, 0, 0, 0], # Pa
  'LambdaNorm_mx'   : np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
  'PiNorm_mx'       : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
  'dNorm_vector': np.array([0, 0, 0, 0, 0, 0]),
  'fNorm_vector': np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
}

constraints = {'separation_factor':2.2}
objective = "molar_power"

pyomo_model = mdl.NFSystem_model(constraints,objective,pm)
pyomo_model, status = initialize.nf_initialization(pyomo_model, process_parameters, pm_sim, pm)
pyomo_model, _, pyomo_optimal = mdl.optimize(pyomo_model,solver='ipopt')

OPTIMAL = pyomo_optimal
DIVALENT_REC = pyomo_model.recovery.value
WATER_REC = pyomo_model.water_recovery.value
SEP_FACTOR = pyomo_model.separation_factor.value
MOL_POWER = pyomo_model.mol_power.value
FEED_PRESSURE = pyomo_model.p_feed.value
PERMEATE_PRESSURES.append(str([float(pyomo_model.stages[k].pp.value) for k in range(pm['nst'])]))
LAMBDA.append(str(pyomo_to_np_array(pyomo_model.Lambda, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
PI.append(str(pyomo_to_np_array(pyomo_model.Pi, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
DILUTION_FRACTIONS.append(str(pyomo_to_np_array(pyomo_model.dNorm, sizes=[pm['nst'],1], dimesions=1).tolist()))
FEED_FRACTIONS.append(str(pyomo_to_np_array(pyomo_model.fNorm, sizes=[pm['nst'],1], dimesions=1).tolist()))
LAMBDA_NORM.append(str(pyomo_to_np_array(pyomo_model.LambdaNorm, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
PI_NORM.append(str(pyomo_to_np_array(pyomo_model.PiNorm, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))

res_data_all = {
  'WATER_REC': WATER_REC,
  'DIVALENT_REC': DIVALENT_REC,
  'SEP_FACTOR': SEP_FACTOR,
  'MOL_POWER': MOL_POWER,
  'FEED_PRESSURE': FEED_PRESSURE,
  'PERMEATE_PRESSURES': PERMEATE_PRESSURES,
  'LAMBDA': LAMBDA,
  'PI': PI,
  'DILUTION_FRACTIONS': DILUTION_FRACTIONS,
  'FEED_FRACTIONS': FEED_FRACTIONS,
  'LAMBDA_NORM': LAMBDA_NORM,
  'PI_NORM': PI_NORM,
  'OPTIMAL': OPTIMAL
}

results_all_df = pd.DataFrame(res_data_all)
results_all_df.to_csv("single_ipopt_example.csv")
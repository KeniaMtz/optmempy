import pandas as pd
import numpy as np
from optmempy.ipopt_framework import opti_initialize as initialize
from optmempy.ipopt_framework import opti_model as mdl
from optmempy.ipopt_framework.utils_ipopt import pyomo_to_np_array
from datetime import datetime

def multiobjective_optimization_multistart(opti_type,
                                          target_folder,
                                          no_of_models,
                                          pm,
                                          pm_sim,
                                          constraint_type,
                                          objective_type,
                                          constraint_levels,
                                          initialization=None,
                                          additional_rec_constraint=0):
  '''
  Multiobjective optimization, multistart.
  Initialization:
  if None: random
  or list of process parameter dictionaries
  '''

  if initialization != None:
    assert len(initialization) == no_of_models, "No. of models should equal no. of initializations."

  now = datetime.now()
  timestamp = now.strftime("%y%m%d_%H%M")
  prefix = f"{timestamp}_opti_{pm['transport_model']}_dpmax_{pm['dp_max']}_T_{pm['T']}_"
  if pm['model_x'] is False:
    prefix = prefix + 'NoPEX_'
  file_name = prefix+opti_type

  length = len(constraint_levels)*no_of_models

  WATER_REC = np.zeros(length)
  DIVALENT_REC = np.zeros(length)
  SEP_FACTOR = np.zeros(length)
  MOL_POWER = np.zeros(length)
  FEED_PRESSURE = np.zeros(length)
  PERMEATE_PRESSURES = []
  LAMBDA = []
  PI = []
  DILUTION_FRACTIONS = []
  FEED_FRACTIONS = []
  LAMBDA_NORM = []
  PI_NORM = []
  OPTIMAL = np.zeros(length)

  i = 0
  for cl in constraint_levels:
    for model_idx in range(no_of_models): 
      print(i)
      if additional_rec_constraint == 0:
        constraints={constraint_type:cl}
      else:
        constraints={constraint_type:cl, 'recovery':additional_rec_constraint}
      objective = objective_type

      pyomo_model = mdl.NFSystem_model(constraints, objective, pm)

      if initialization == None:
        pyomo_model, init_parameters = initialize.random_initialization(pyomo_model,pm_sim,pm)
      else:
        pyomo_model, status = initialize.nf_initialization(pyomo_model, initialization[model_idx], pm_sim, pm)

      pyomo_model, _, pyomo_optimal = mdl.optimize(pyomo_model,solver='ipopt')

      OPTIMAL[i] = pyomo_optimal
      DIVALENT_REC[i] = pyomo_model.recovery.value
      WATER_REC[i] = pyomo_model.water_recovery.value
      SEP_FACTOR[i] = pyomo_model.separation_factor.value
      MOL_POWER[i] = pyomo_model.mol_power.value
      FEED_PRESSURE[i] = pyomo_model.p_feed.value
      PERMEATE_PRESSURES.append(str([float(pyomo_model.stages[k].pp.value) for k in range(pm['nst'])]))
      LAMBDA.append(str(pyomo_to_np_array(pyomo_model.Lambda, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
      PI.append(str(pyomo_to_np_array(pyomo_model.Pi, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
      DILUTION_FRACTIONS.append(str(pyomo_to_np_array(pyomo_model.dNorm, sizes=[pm['nst'],1], dimesions=1).tolist()))
      FEED_FRACTIONS.append(str(pyomo_to_np_array(pyomo_model.fNorm, sizes=[pm['nst'],1], dimesions=1).tolist()))
      LAMBDA_NORM.append(str(pyomo_to_np_array(pyomo_model.LambdaNorm, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
      PI_NORM.append(str(pyomo_to_np_array(pyomo_model.PiNorm, sizes=[pm['nst'],pm['nst']], dimesions=2).tolist()))
  
      i += 1

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

  results_all_df.to_csv(target_folder / f"{file_name}.csv")

  return file_name

def pareto_selector(file_name_without_csv,objective1,objective2,competing='true'):
  '''
  competing: 
  true -> classical pareto trade-off
  1min -> minimizing objective 1, maximizing objective 2
  2min -> minimizing objective 2, maximizing objective 1
  '''

  input_file_path = f"{file_name_without_csv}.csv"
  output_file_path = f"{file_name_without_csv}_pareto.csv"

  df = pd.read_csv(input_file_path)

  pareto_optimal = []
  pareto_optimal_points = []

  for index, row in df.iterrows():
    current_point = (round(row[objective1],4), round(row[objective2],4))
    is_pareto_optimal = True
    is_feasible = True

    for idx, rw in df.iterrows():
      current_point2 = (round(rw[objective1],4), round(rw[objective2],4))
      if competing == 'true':
        if (current_point2[0] >= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
          is_pareto_optimal = False
          break
      elif competing == '1min':
        if (current_point2[0] <= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
          is_pareto_optimal = False
          break
      elif competing == '2min':
        if (current_point2[0] >= current_point[0] and current_point2[1] <= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
          is_pareto_optimal = False
          break
    if is_pareto_optimal and row['OPTIMAL'] == 1 and is_feasible:
      pareto_optimal.append(row)
      pareto_optimal_points.append(current_point)

  pareto_df = pd.DataFrame(pareto_optimal)

  pareto_df.to_csv(output_file_path, index=False)
from optmempy.ipopt_framework import opti_multiobjective
from optmempy.ipopt_framework import opti_initialize
from optmempy.ipopt_framework.utils_ipopt import (
  read_input,
  check_input_data,
  read_initialization_file,
)
import numpy as np

if __name__ == "__main__":

  input_path = input('Input file path or name: ')
  if not input_path.endswith(".txt"):
    input_path += ".txt"

  try:
    data = read_input(input_path)
  except FileNotFoundError:
    print('File not found. All input data has to be provided in an input .txt file.')
    exit()

  is_error, error_message = check_input_data(data)
  if is_error:
    print(error_message)
    exit()

  no_of_models = data['no_models']
  dp_max = data['max_pressure']
  objective_type = data['objective']
  n_stages = data['n_stages']
  transport_model = data['transport_model']
  T = data['temperature']
  constraint_levels = data['constraint_list']

  if 'initialization_file' in data:
    print('\nUsing user-defined initialization.')
    try:
      initialization = read_initialization_file(csv_file=data['initialization_file'], n_stages=n_stages, dp_max=dp_max)
      if len(initialization) != no_of_models:
        raise ValueError(
          'Number of rows in initialization '
          'file must match no_models.'
        )
    except FileNotFoundError:
      print('File not found. Initialization data has to be provided in an input .csv file.')
      exit()
  else:
    initialization = None

  if objective_type == 'separation_factor':
    constraint_type = 'recovery'
    opti_type = 'obj_sf_const_rec'
  elif objective_type == 'molar_power':
    constraint_type = 'separation_factor'
    opti_type = 'obj_mp_const_sf'

  additional_rec_constraint = 0

  if data['pressure_exchange'] == 1:
    model_x = True
  elif data['pressure_exchange'] == 0:
    model_x = False

  if data['relax'] == 1:
    relax = True
  elif data['relax'] == 0:
    relax = False

  from pathlib import Path
  import optmempy.ipopt_framework as ipopt_framework
  BASE_DIR = Path(ipopt_framework.__file__).resolve().parent
  target_folder = BASE_DIR / "results"
  target_folder.mkdir(exist_ok=True)

  print("STARTING MULTISTART OPTIMIZATION\n")
  pm, pm_sim = opti_initialize.load_problem(n_stages=n_stages,
                               transport_model=transport_model,
                               relax=relax,
                               dp_max=dp_max,
                               model_x=model_x,
                               T = T)
  file_name = opti_multiobjective.multiobjective_optimization_multistart(opti_type=opti_type,
                                                             target_folder=target_folder,
                                                             no_of_models=no_of_models,
                                                             pm=pm,
                                                             pm_sim=pm_sim,
                                                             constraint_type=constraint_type,
                                                             objective_type=objective_type,
                                                             constraint_levels=constraint_levels,
                                                             initialization=initialization,
                                                             additional_rec_constraint=additional_rec_constraint)

  print("STARTING VALIDATION THROUGH PROCESS SIMULATION\n")
  input_file = target_folder / file_name
  opti_initialize.sim_validation(input_file, pm_sim)

  print("STARTING PARETO SELECTION\n")
  input_file = target_folder / f"{file_name}_validation"

  if objective_type == 'separation_factor':
    opti_multiobjective.pareto_selector(input_file, 'DIVALENT_REC', 'SEP_FACTOR', competing='true')
  if objective_type == 'molar_power':
    opti_multiobjective.pareto_selector(input_file, 'SEP_FACTOR', 'MOL_POWER', competing='2min')

  print("TERMINATION SUCCESSFUL")
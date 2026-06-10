import numpy as np
import pandas as pd

def pyomo_to_np_array(pyomo_variable, sizes = [3,3], dimesions = 2):
  """
  Convert a Pyomo variable to a NumPy array.
  :param pyomo_variable: The Pyomo variable to convert.
  :param sizes: The sizes of the dimensions of the NumPy array.
  :param dimesions: The number of dimensions of the NumPy array.
  :return: A NumPy array with the same values as the Pyomo variable.
  """
  if dimesions == 1:
    return np.array([float(round(pyomo_variable[i].value,2)) for i in range(sizes[0])])
  elif dimesions == 2:
    return np.array([[float(round(pyomo_variable[i,j].value,2)) for j in range(sizes[1])] for i in range(sizes[0])])
  

def read_input(input_file_path):
  '''
  Reads inputs provided in .txt file.
  '''
  data_dict = {}
  with open(input_file_path, 'r') as file:
    for line in file:
      key, value = line.strip().split('=', 1)
      try:
        value = float(value)
      except ValueError:
        pass
      data_dict[key] = value
  
  data_dict['constraint_list'] = []
  for c in list(data_dict['constraints'].split(",")):
    data_dict['constraint_list'].append(float(c))
  data_dict['pressure_exchange'] = int(data_dict['pressure_exchange']) # 0 or 1. 0: not included, 1: included.
  data_dict['no_models'] = int(data_dict['no_models']) # Number of random initiations for multistart optimization.
  
  data_dict['n_stages'] = int(data_dict['n_stages'])
  data_dict['relax'] = int(data_dict['relax']) # 0 or 1. 1: True -> NLP, 0: False -> MINLP.
  data_dict['temperature'] = float(data_dict['temperature'])
  
  return data_dict

def check_input_data(data_dict):
  error = False
  error_message = ''
  if data_dict['objective'] not in ('separation_factor','molar_power'):
    error = True
    error_message = 'Unknown objective.'
    return error, error_message
  
  if data_dict['transport_model'] != 'sdec':
    error = True
    error_message = 'Unknown transport model'

  if data_dict['max_pressure'] > 50 or data_dict['max_pressure'] < 10:
    error = True
    error_message = 'Feed pressure has to be between 10 and 50 barg.'
    return error, error_message
  
  if data_dict['temperature'] > 321.0 or data_dict['temperature'] < 293.0:
    error = True
    error_message = 'Temperature has to be between 293 and 321 K.'
  
  if data_dict['no_models'] > 20 or data_dict['no_models'] < 1:
    error = True
    error_message = 'Number of models has to be between 1 and 20.'
    return error, error_message
  
  if data_dict['n_stages'] > 10 or data_dict['n_stages'] < 2:
    error = True
    error_message = 'Number of stages has to be betweem 2 and 10'
  
  if data_dict['objective'] == 'separation_factor':
    try:
      for const in data_dict['constraint_list']:
        if const < 0.0 or const > 1.0:
          error = True
          error_message = 'Recovery constraint for separation factor optimization has to be between 0 and 1.'
          return error, error_message
    except:
      error = True
      error_message = 'Constraint error.'
      return error, error_message

  if data_dict['objective'] == 'molar_power':
    try:
      for const in data_dict['constraint_list']:
        if const < 1.0 or const > 10.0:
          error = True
          error_message = 'Separation factor constraint for molar power optimization has to be between 1 and 10.'
          return error, error_message
    except:
      error = True
      error_message = 'Constraint error.'
      return error, error_message                   

  if data_dict['pressure_exchange'] != 1 and data_dict['pressure_exchange'] != 0:
    error = True
    error_message = 'Pressure exchange value has to be 1 or 0.'
    return error, error_message      

  if data_dict['relax'] != 1 and data_dict['relax'] != 0:
    error = True
    error_message = 'relax value has to be 1 or 0.'
    return error, error_message                       

  return error, error_message

def read_initialization_file(csv_file, n_stages, dp_max):
  """
  Reads initialization CSV file and converts
  each row into a process_parameters dictionary.
  """
  file_name = str(csv_file)
  if not file_name.endswith(".csv"):
    file_name += ".csv"

  df = pd.read_csv(file_name)

  initialization_list = []

  for n, row in df.iterrows():
    n += 1
    process_parameters = {}

    process_parameters['p_feed'] = (row['p_feed'] * 1e5)
    process_parameters['pp_list'] = np.array([row[f'pp_{i}'] for i in range(n_stages)]) * 1e5
    process_parameters['LambdaNorm_mx'] = np.array([[row[f'LambdaNorm_{i}_{j}'] for j in range(n_stages)] for i in range(n_stages)])
    process_parameters['PiNorm_mx'] = np.array([[row[f'PiNorm_{i}_{j}'] for j in range(n_stages)] for i in range(n_stages)])
    process_parameters['dNorm_vector'] = np.array([row[f'dNorm_{i}'] for i in range(n_stages)])
    process_parameters['fNorm_vector'] = np.array([row[f'fNorm_{i}'] for i in range(n_stages)])

    p_feed_bar = (process_parameters["p_feed"] / 1e5)

    if p_feed_bar < 10 or p_feed_bar > dp_max:
      raise ValueError(
        f"[Row {n}] "
        f"p_feed must be between "
        f"10 and {dp_max} bar."
      )

    pp_list_bar = (process_parameters["pp_list"] / 1e5)

    for i, pp in enumerate(pp_list_bar):
      if pp < 0:
        raise ValueError(
          f"[Row {n}] "
          f"pp_{i} cannot be negative."
        )

      if pp > dp_max:
        raise ValueError(
          f"[Row {n}] "
          f"pp_{i} exceeds dp_max."
        )

      if pp > p_feed_bar:
        raise ValueError(
          f"[Row {n}] "
          f"pp_{i} exceeds p_feed."
        )

      if pp > 0.8 * p_feed_bar:
        raise ValueError(
          f"[Row {n}] "
          f"pp_{i} exceeds "
          f"0.8 * p_feed."
        )

    Lambda = process_parameters["LambdaNorm_mx"]

    for i in range(n_stages):
      row_sum = np.sum(Lambda[i])

      if row_sum > 1 + 1e-12:
        raise ValueError(
          f"[Row {n}] "
          f"Lambda row {i} "
          f"sums to more than 1."
        )

      for j in range(n_stages):
        val = Lambda[i, j]
        if i == j and val != 0:
          raise ValueError(
            f"[Row {n}] "
            f"LambdaNorm_{i}_{j} "
            f"must be 0 "
            f"(no self-recirculation)."
          )
        
        if val < 0 or val > 1:
          raise ValueError(
            f"[Row {n}] "
            f"LambdaNorm_{i}_{j} "
            f"must be between 0 and 1."
            )

    Pi = process_parameters["PiNorm_mx"]

    for i in range(n_stages):
      row_sum = np.sum(Pi[i])

      if row_sum > 1 + 1e-12:
        raise ValueError(
          f"[Row {n}] "
          f"Pi row {i} "
          f"sums to more than 1."
        )

      for j in range(n_stages):
        val = Pi[i, j]
        if i == j and val != 0:
          raise ValueError(
            f"[Row {n}] "
            f"PiNorm_{i}_{j} "
            f"must be 0 "
            f"(no self-recirculation)."
          )
        
        if val < 0 or val > 1:
          raise ValueError(
            f"[Row {n}] "
            f"PiNorm_{i}_{j} "
            f"must be between 0 and 1."
            )

    dNorm = process_parameters["dNorm_vector"]

    for i, val in enumerate(dNorm):
      if val < 0 or val > 1:
        raise ValueError(
          f"[Row {n}] "
          f"dNorm_{i} must be "
          f"between 0 and 1."
          )

    if np.sum(dNorm) > 1 + 1e-12:
      raise ValueError(
        f"[Row {n}] "
        f"dNorm_vector sums "
        f"to more than 1."
      )

    fNorm = process_parameters["fNorm_vector"]

    for i, val in enumerate(fNorm):
      if val < 0 or val > 1:
        raise ValueError(
          f"[Row {n}] "
          f"fNorm_{i} must be "
          f"between 0 and 1."
          )

    if abs(np.sum(fNorm) - 1) > 1e-12:
      raise ValueError(
        f"[Row {n}]"
        f"fNorm_vector must sum to 1."
      )

    initialization_list.append(process_parameters)

  return initialization_list
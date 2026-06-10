import numpy as np
import re

def ffvar_to_float(v):
    s = str(v)
    match = re.search(r"value ([0-9eE+\-.]+)", s)
    return float(match.group(1))

def scale_x(x_real, x_scaler, scale_idx):
    x_real = np.array(x_real, dtype=float)
    x_scaled = x_real.copy()

    x_scaled_subset = x_scaler.transform(x_real[scale_idx].reshape(1, -1))

    x_scaled[scale_idx] = x_scaled_subset.flatten()
    return x_scaled

def unscale_x(x_scaled, x_scaler, scale_idx):
    x_scaled = np.array(x_scaled, dtype=float)
    x_real = x_scaled.copy()

    x_real_subset = x_scaler.inverse_transform(
        x_scaled[scale_idx].reshape(1, -1)
    )

    x_real[scale_idx] = x_real_subset.flatten()
    return x_real

def scale_y(y_real, y_scaler):
    y_real = np.array(y_real, dtype=float).reshape(1, -1)
    return y_scaler.transform(y_real).flatten()

def unscale_y(y_scaled, y_scaler):
    y_scaled = np.array(y_scaled, dtype=float).reshape(1, -1)
    return y_scaler.inverse_transform(y_scaled).flatten()

def scale_single_value(value, idx, scaler):
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    return (value - data_min[idx]) / (data_max[idx] - data_min[idx])

def unscale_single_value(value, idx, scaler):
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    return value * (data_max[idx] - data_min[idx]) + data_min[idx]

def get_ann_input_vector(pm, pp, expected_length=8):
    ns = pm['nst']

    data_map = {
    'T': pm['T'],
    'p_feed': pp['p_feed']
    }

    for i in range(ns):
        data_map[f"pp_{i}"] = pp['pp_list'][i]

    input_scale_cols = (
        ["T", "p_feed"] +
        [f"pp_{i}" for i in range(ns)]
        )
    input_cols = input_scale_cols
    try:
        x_vector = np.array([data_map[col] for col in input_cols])

        if x_vector.shape[0] != expected_length:
            raise ValueError(f"Shape mismatch: expected {expected_length}, got {x_vector.shape[0]}")

        #print(f"x vector shape is correct: {x_vector.shape[0]}") 
        return x_vector

    except KeyError as e:
        raise KeyError(f"Missing parameter in dictionaries: {e}")
    
def get_idx_map(ns):
    n_fixed = 1         # Constants during optimization: T
    n_pf = 1            # p_feed
    n_pp = ns           # pp_list

    idx = {}

    start = 0
    idx['fixed'] = (start, start + n_fixed)
    start += n_fixed

    idx['p_feed'] =  (start, start + n_pf)
    start += n_pf
    
    idx['pp'] = (start, start + n_pp)
    start += n_pp

    return idx


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
  
  data_dict['n_stages'] = int(data_dict['n_stages'])
  data_dict['temperature'] = float(data_dict['temperature'])
  data_dict['plot_pareto'] = int(data_dict['plot_pareto']) # 0 or 1. 0: not included, 1: included.
  
  return data_dict

def check_input_data(data_dict):
  error = False
  error_message = ''
  if data_dict['objective'] not in ('separation_factor','molar_power'):
    error = True
    error_message = 'Unknown objective.'
    return error, error_message

  if data_dict['max_pressure'] > 41 or data_dict['max_pressure'] < 10:
    error = True
    error_message = 'Feed pressure has to be between 10 and 40 barg.'
    return error, error_message
  
  if data_dict['temperature'] > 321.0 or data_dict['temperature'] < 293.0:
    error = True
    error_message = 'Temperature has to be between 293 and 321 K.'
  
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

  return error, error_message
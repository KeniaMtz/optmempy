from optmempy.maingo_framework.run_maingo import run_and_validate
from optmempy.maingo_framework.utils import read_input, check_input_data
import time
import numpy as np
import pandas as pd
from datetime import datetime

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

  dp_max = data['max_pressure']
  objective_type = data['objective']
  n_stages = data['n_stages']
  T = data['temperature']
  constraint_list = data['constraint_list']
  
  if objective_type == 'separation_factor':
    constraint_type = 'recovery'
    opti_type = 'obj_sf_const_rec'
  elif objective_type == 'molar_power':
    constraint_type = 'separation_factor'
    opti_type = 'obj_mp_const_sf'

  now = datetime.now()
  timestamp = now.strftime("%y%m%d_%H%M")
  prefix = f"{timestamp}_dpmax_{dp_max}_bar_T_{T}_K_"
  file_name = prefix + opti_type

  if data['plot_pareto'] == 1:
    plot = True
  elif data['plot_pareto'] == 0:
    plot = False

  # NF cascade system process parameters.
  # pm and pp dictionaries MUST match the configuration trained on the ANN!

  pm_sim = {
    'F_feed'     : 20.0, # m3/h
    'F_dil_feed' : 10.0, # m3/h
    'T'          : T, # K
    'c_feed'     : [10.925, 60, 600.173913, 10.46153846, 31.38541667, 685.6901408], # mol/m3
    'ns'         : 6,
    'ne'         : 3,
    'nst'        : n_stages,
    'model_x'    : True,
    'pex_eff'    : 0.85,
    'pump_eff'   : 0.85,
    'dp_max'     : dp_max,
  }

  process_parameters = {
    'p_feed'          :  29.7255,  # bar - Will be overwritten
    'pp_list'         :  [0, 0, 0, 0, 0, 0],  # bar - Will be overwritten
    'LambdaNorm_mx'   : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.31, 0.0, 0.0, 0.69, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    'PiNorm_mx'       : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.01, 0.0, 0.16, 0.39],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    'dNorm_vector': np.array([0.43, 0.43, 0.02, 0.12, 0.0, 0.0]),
    'fNorm_vector': np.array([0.0, 0.0, 0.37, 0.0, 0.34, 0.28]),
  }

  results = []
  for i, cval in enumerate(constraint_list):
    print(f"\n[{i+1}/{len(constraint_list)}] Constraint = {cval:.4f}")
    start_cpu = time.process_time()
    try:
      res = run_and_validate(pm_sim, process_parameters, objective_type=objective_type, constraint_value=cval, VERBOSITY=False)
      elapsed_cpu = time.process_time() - start_cpu
      res["constraint_value"] = cval
      res["cpu_time_sec"] = elapsed_cpu
      results.append(res)
    except Exception or NameError as e:
      print(f'FAILED!! with constraint {cval}. {e}')
      continue

  if len(results) == 0:
    raise RuntimeError("No successful optimization runs.")

  from pathlib import Path
  import optmempy.maingo_framework as maingo_framework
  BASE_DIR = Path(maingo_framework.__file__).resolve().parent
  target_folder = BASE_DIR / "results"
  target_folder.mkdir(exist_ok=True)
  
  df = pd.DataFrame(results)
  df.to_csv(target_folder / f"{file_name}.csv", index=False)
  print("\nSaved:", file_name)

  if plot:
    print("\n> Plotting pareto plot")
    import matplotlib.pyplot as plt
    x = df["separation_factor"]
    y = df["molar_power"]

    plt.figure(figsize=(7,5))
    plt.plot(x, y, "o-", linewidth=2)
    plt.xlabel("Separation factor")
    plt.ylabel("Specific molar power")
    plt.title("Pareto curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

  print("\nOptimization run finished.")
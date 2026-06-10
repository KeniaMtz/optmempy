from optmempy.maingo_framework.run_maingo import run_and_validate
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pm and pp should match how the ANN was trained.
# Inputs
pm_sim = {
  'F_feed'     : 20.0, # m3/h
  'F_dil_feed' : 10.0, # m3/h
  'T'          : 304.15, # K
  'c_feed'     : [10.925, 60, 600.173913, 10.46153846, 31.38541667, 685.6901408], # mol/m3
  'ns'         : 6,
  'ne'         : 3,
  'nst'        : 6,
  'model_x'    : True,
  'pex_eff'    : 0.85,
  'pump_eff'   : 0.85,
  'dp_max'     : 40,
}

process_parameters = {
  'p_feed'          :  29.7255,  # bar
  'pp_list'         :  [0, 0, 0, 0, 0, 0],  # bar
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

#constraint_list = np.arange(4, 10, 0.5)
#print(constraint_list)
constraint_list = [0.93, 0.94]

results = []
for i, cval in enumerate(constraint_list):
  print(f"\n[{i+1}/{len(constraint_list)}] Constraint = {cval:.4f}")
  start_cpu = time.process_time()
  try:
    res = run_and_validate(pm_sim, process_parameters, objective_type="separation_factor", constraint_value=cval, VERBOSITY=True)
    elapsed_cpu = time.process_time() - start_cpu
    res["constraint_value"] = cval
    res["cpu_time_sec"] = elapsed_cpu
    results.append(res)
  except Exception or NameError as e:
    print(f'FAILED!! with constraint {cval}. {e}')
    continue

if len(results) == 0:
  raise RuntimeError("No successful optimization runs.")

print(results)

df = pd.DataFrame(results)
filename = f"pareto_sepfactor_304.15_K.csv"
df.to_csv(filename, index=False)
print("Saved:", filename)

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
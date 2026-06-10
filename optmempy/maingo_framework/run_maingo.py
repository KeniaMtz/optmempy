import maingopy
import numpy as np

from optmempy.maingo_framework.utils import (
  get_ann_input_vector,
  scale_x,
  unscale_y,
  unscale_single_value,
  ffvar_to_float,
)
from optmempy.maingo_framework.problem import NF_MAiNGO_Model
from optmempy.nf_cascade_simulation import NF_simulation as simu


def run_one_constraint(pm, pp, objective_type, constraint_value, VERBOSITY=False):
  _ = get_ann_input_vector(pm, pp)

  model = NF_MAiNGO_Model(pm, pp, objective_type=objective_type, constraint_value=constraint_value)

  solver = maingopy.MAiNGO(model)

  solver.set_option("epsilonA", 5e-2)
  solver.set_option("epsilonR", 5e-2)
  solver.set_option("maxwTime", 1e500)
  solver.set_option("maxTime", 86400) # 1 day in seconds
  solver.set_option("writeResultFile", 0)
  solver.set_option("loggingDestination", 1)

  if not VERBOSITY:
    solver.set_option("BAB_verbosity", 0)
    solver.set_option("UBP_verbosity", 0)
    solver.set_option("LBP_verbosity", 0)
    solver.set_option("OUTSTREAMVERBOSITY", 0)
    solver.set_option("PRE_printEveryLocalSearch", False)

  print("\nInitiating MAiNGO...")
  status = solver.solve()
  print("\nOptimization done...")

  sol = solver.get_solution_point()
  obj_value = solver.get_objective_value()

  if objective_type == "separation_factor":
    sep_scaled = -obj_value
    sep_real = unscale_single_value(sep_scaled, 1, model.y_scaler)

  elif objective_type == "molar_power":
    mpower_scaled = obj_value
    mpower_log = unscale_single_value(mpower_scaled, 2, model.y_scaler)
    mpower_real = np.expm1(mpower_log)

  x_scaled = list(sol)

  NST = pm["nst"]

  input_names = (["T", "p_feed"] + [f"pp_{i}" for i in range(NST)])
  x_scale_cols = (["p_feed"] +[f"pp_{i}" for i in range(NST)])
  x_idx = [input_names.index(c) for c in x_scale_cols]
  x_real = x_scaled.copy()

  for i, val in enumerate(x_idx):
    x_real[i] = unscale_single_value(x_scaled[i], val, model.x_scaler)

  ns = pm["nst"]
  start = 0
  p_feed = x_real[start]
  start += 1
  pp_list = x_real[start:start + ns]
  start += ns

  pp_maingo = {
    "p_feed": p_feed,
    "pp_list": pp_list,
  }

  x_vector_real = get_ann_input_vector(pm, pp_maingo)
  x_vector_scaled = scale_x(x_vector_real, model.x_scaler, model.scale_idx)
  y_scaled = model.ann_model.predict(x_vector_scaled)
  y_scaled = np.array([ffvar_to_float(v) for v in y_scaled])
  y_real = unscale_y(y_scaled, model.y_scaler)
  y_real[2] = np.expm1(y_real[2])

  keys = ["recovery", "separation_factor", "molar_power"]
  if len(y_real) == len(keys):
    ann_pred = dict(zip(keys, y_real))
  else:
    raise ValueError(f"Expected {len(keys)} values, "f"but got {len(y_real)}")

  if objective_type == "separation_factor":
    objective_real = ann_pred["separation_factor"]
    objective_maingo_real = sep_real
  else:
    objective_real = ann_pred["molar_power"]
    objective_maingo_real = mpower_real

  return {
    "status": status,
    "objective_real": objective_real,
    "objective_maingo_real": objective_maingo_real,
    "objective_maingo_scaled": obj_value,
    "solution_point_real": x_real,
    "solution_point_scaled": x_scaled,
    "recovery": ann_pred["recovery"],
    "separation_factor": ann_pred["separation_factor"],
    "molar_power": ann_pred["molar_power"],
  }


def check_feasibility(desc, sol_dict, n_stages, n_solutes, n_elements):
  init_error = False

  for i in range(n_stages):
    if desc['Psi_d'][i] < 0:
      init_error = True
      print('Negative dil power')
      break
    for j in range(n_stages):
      if desc['Psi_Lambda'][i,j] < 0:
        init_error = True
        print('Negative Lambda power')
        break

  for idx, el in enumerate(desc.values()):
    try:
      if el < 0:
        init_error = True
        print('Negative characteristic')
        print(idx, el)
        break
    except (TypeError,ValueError):
      pass
    try:
      if np.any(el<0):
        init_error = True
        print('Negative characteristic')
        print(idx, el)
        break
    except (TypeError,ValueError):
      pass

  for i in range(n_stages):
    for d in ['Fr','Fp','F0','p0','pr','pp']:
      if sol_dict[i].simulation[d] < 0:
        init_error = True
        print('Negative flow or pressure')
        break
    for d in ['c0','cr','cp']:
      for j in range(n_solutes):
        if sol_dict[i].simulation[d][j] < 0:
          init_error = True
          print('Negative concentration',"Stage",i,d)
          continue#break
    
    for el in range(n_elements):
      if sol_dict[i].simulation['elements'][el]['p0'] - sol_dict[i].simulation['elements'][el]['pp'] < 0:
        init_error = True
        break
      for d in ['Fr','Fp','F0','p0','pr','pp']:
        if sol_dict[i].simulation['elements'][el][d] < 0:
          init_error = True
          print('Negative flow or pressure')
          break
      for d in ['c0','cr','cp']:
        for j in range(n_solutes):
          if sol_dict[i].simulation['elements'][el][d][j] < 0:
            init_error = True
            print('Negative concentration',"Stage",i,"element",el,d,"ion",j)
            continue#break

  return init_error


def validate_solution(solution_point_real, pm, pp):
  p_feed_val = solution_point_real[0] * 1e5
  pp_list_vals = (np.array(solution_point_real[1:1 + pm["nst"]]) * 1e5)

  process_parameters = (pp.copy())
  process_parameters["p_feed"] = p_feed_val
  process_parameters["pp_list"] = pp_list_vals

  sim_cascade = simu.NFcascade(pm, process_parameters)
  sim_cascade.simulate()
  desc, sol_dict = (sim_cascade.compute_descriptors())

  infeasibility = check_feasibility(desc, sol_dict, pm["nst"], n_solutes=6, n_elements=3)

  molar_power_kW = desc["molar_power"]
  molar_power_kW = ((molar_power_kW / 100000) * 100000 / (3600 * 1000))

  return {
    "SIM_recovery": desc["recovery"],
    "SIM_separation_factor": (desc["separation_factor"]),
    "SIM_molar_power": molar_power_kW,
    "Infeasible": infeasibility
  }

def run_and_validate(
  pm,
  pp,
  objective_type,
  constraint_value,
  VERBOSITY=False,
):
  result = run_one_constraint(
    pm=pm,
    pp=pp,
    objective_type=objective_type,
    constraint_value=constraint_value,
    VERBOSITY=VERBOSITY,
  )
  print("\n> Run one constraint: Success.")

  print("\n> Starting simulation validation.")
  validation = validate_solution(
    solution_point_real=(result["solution_point_real"]),
    pm=pm,
    pp=(pp)
  )

  result.update(validation)
  print("\nDONE!")

  return result
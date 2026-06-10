from optmempy.nf_cascade_simulation import NF_simulation as nf_simulation
import numpy as np
import pandas as pd
import warnings
import ast

def load_problem(n_stages: int = 4,
                 transport_model: str = 'sdec',
                 relax: bool = True, # Problem relaxation. If True -> NLP, if False -> MINLP
                 dp_max: float = 41.0, # bar
                 model_x: bool = False,
                 T: float = 303.0, # K
                 ):
  

  pm = {
    'F_feed'     : 20.0, # m3/h
    'F_dil_feed' : 10.0, # m3/h
    'T'          : T, # K
    'c_feed'     : [10.925, 60, 600.173913, 10.46153846, 31.38541667, 685.6901408], # mol/m3
    'ne'         : 3,
    'nst'        : n_stages,
    'model_x'    : model_x,
    'pex_eff'    : 0.85,
    'pump_eff'   : 0.85,
    'dp_max'     : dp_max # bar
  }

  pm['relax'] = relax
  pm['transport_model'] = transport_model
  pm['F_lim'] = pm['F_feed'] + pm['F_dil_feed']
  
  pm['c_feed_dict'] = {}
  for i in range(len(pm['c_feed'])):
    pm['c_feed_dict'][i] = pm['c_feed'][i]
  
  pm['ns'] = len(pm['c_feed'])

  pm['CF_max'] = 20

  pm_sim = pm.copy()
  pm_sim['dp_max'] = pm_sim['dp_max'] * 1e5 # Pa

  return pm, pm_sim


def random_initialization(model, pm_sim, pm):
  feasible = 0
  warnings.simplefilter('error', RuntimeWarning)

  def random_upper_triangular(n, max_row_sum=0.9, seed=None):
    if seed is not None:
      np.random.seed(seed)
        
    A = np.zeros((n, n))
    
    for i in range(n):
      k = n - 1 - i
      if k > 0:
        vals = np.random.rand(k)
        row_sum = np.sum(vals)
        
        if row_sum > 0:
          scale_factor = (np.random.rand() * max_row_sum) / row_sum
          vals = vals * scale_factor
        
        A[i, i+1:] = vals
    
    return A

  while feasible == 0:
    try:
      p_feed = np.random.uniform(pm['dp_max']*0.8, pm['dp_max'])*1e5 # Pa
      pp_list = list(np.random.uniform((p_feed/1e5)*0.8, p_feed/1e5,size=(pm_sim['nst']))*1e5) # Pa
      f_vector = np.ones(pm_sim['nst']) * (1/pm_sim['nst'])

      process_parameters = {
        'p_feed': p_feed,
        'pp_list': pp_list,
        'LambdaNorm_mx': random_upper_triangular(pm_sim['nst']),
        'PiNorm_mx': random_upper_triangular(pm_sim['nst']),
        'dNorm_vector': np.random.uniform(0,0.5*(1/pm_sim['nst']),size=(pm_sim['nst'])),
        'fNorm_vector': f_vector,
      }

      print('Initializing with random parameters')
      model, status = nf_initialization(model, process_parameters, pm_sim, pm)
    except RuntimeWarning:
      status = True
    
    if status:
      feasible = 0
    else:
      feasible = 1
  
  return model, process_parameters


def reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements):
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
          break
    
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
            break

  return init_error


def nf_initialization(model, process_parameters, pm_sim, pm):
  status = False
  model_x = pm_sim['model_x']
  transport_model = pm_sim['transport_model']
  n_stages = pm_sim['nst']
  n_solutes = pm_sim['ns']
  n_elements = pm_sim['ne']

  try:
    nf_system = nf_simulation.NFcascade(pm_sim, process_parameters)
    nf_system.simulate()
    desc, sol_dict = nf_system.compute_descriptors()
    print('Simulation done')
  except ValueError as e:
    print("Initialization failed due to ValueError:", e)
    status = True
    return model, status
  
  status = reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements)
  if status:
    return model, status
  print("Feasible model found")

  model.recovery.value = desc['recovery']
  model.separation_factor.value = desc['separation_factor']
  model.water_recovery.value = desc['water_recovery']
  
  model.dilution_ratio.value = desc['dilution_ratio']
  model.p_dil_pump.value = desc['p_dil_pump'] /1e5

  if model_x == False:
    model.p_feed.value = process_parameters['p_feed'] /1e5
  else:
    model.p_pump.value = desc['p_pump'] /1e5
    model.p_feed.value = process_parameters['p_feed'] /1e5

  model.power.value = desc['total_power'] /1e5
  model.mol_power.value = desc['molar_power'] /1e5
  
  for i in range(n_stages):
    model.d[i].value = process_parameters['dNorm_vector'][i] * pm_sim['F_dil_feed']
    model.dNorm[i].value = process_parameters['dNorm_vector'][i]
    model.f[i].value = process_parameters['fNorm_vector'][i] * pm_sim['F_feed']
    model.fNorm[i].value = process_parameters['fNorm_vector'][i]
    model.Psi_d[i].value = desc['Psi_d'][i] / 1e5

  for i in range(n_stages):
    for j in range(n_stages):
      model.Lambda[i,j].value = desc['Lambda_mx'][i,j]
      model.LambdaNorm[i,j].value = process_parameters['LambdaNorm_mx'][i,j]
      model.Pi[i,j].value = desc['Pi_mx'][i,j]
      model.PiNorm[i,j].value = process_parameters['PiNorm_mx'][i,j]
      model.Psi_Lambda[i,j].value = desc['Psi_Lambda'][i,j] / 1e5
      model.Psi_Pi[i,j].value = desc['Psi_Pi'][i,j] / 1e5

  model.retentate_flow_rate.value = desc['retentate_flow_rate']
  model.permeate_flow_rate.value = desc['permeate_flow_rate']

  model.retentate_pressure.value = desc['retentate_pressure'] /1e5

  for i in range(n_solutes):
    model.retentate_concentration[i].value = desc['c_retentate'][i]
    model.permeate_concentration[i].value = desc['c_permeate'][i]

  for st in range(n_stages):
    model.stages[st].p0.value = sol_dict[st].simulation['p0'] /1e5
    model.stages[st].pp.value = sol_dict[st].simulation['pp'] /1e5
    model.stages[st].pr.value = sol_dict[st].simulation['pr'] /1e5
    model.stages[st].F0.value = sol_dict[st].simulation['F0']
    model.stages[st].Fr.value = sol_dict[st].simulation['Fr']
    model.stages[st].Fp.value = sol_dict[st].simulation['Fp']
    
    for i in range(n_solutes):
      model.stages[st].C0[i].value = sol_dict[st].simulation['c0'][i]
      model.stages[st].Cr[i].value = sol_dict[st].simulation['cr'][i]
      model.stages[st].Cp[i].value = sol_dict[st].simulation['cp'][i]

    for el in range(pm_sim['ne']):
      # element-level
      model.stages[st].elems[el].p0.value = sol_dict[st].simulation['elements'][el]['p0'] /1e5 # bar
      model.stages[st].elems[el].pp.value = sol_dict[st].simulation['elements'][el]['pp'] /1e5 # bar
      model.stages[st].elems[el].pr.value = sol_dict[st].simulation['elements'][el]['pr'] /1e5 # bar

      model.stages[st].elems[el].F0.value = sol_dict[st].simulation['elements'][el]['F0']
      model.stages[st].elems[el].Fr.value = sol_dict[st].simulation['elements'][el]['Fr']
      model.stages[st].elems[el].Fp.value = sol_dict[st].simulation['elements'][el]['Fp']
      
      for i in range(n_solutes):
        model.stages[st].elems[el].C0[i].value = sol_dict[st].simulation['elements'][el]['c0'][i]
        model.stages[st].elems[el].Cr[i].value = sol_dict[st].simulation['elements'][el]['cr'][i]
        model.stages[st].elems[el].Cp[i].value = sol_dict[st].simulation['elements'][el]['cp'][i]

      model.stages[st].elems[el].ann.F0.value = model.stages[st].elems[el].F0.value
      model.stages[st].elems[el].ann.p0.value = model.stages[st].elems[el].p0.value
      model.stages[st].elems[el].ann.pp.value = model.stages[st].elems[el].pp.value
      for i in range(n_solutes):
        model.stages[st].elems[el].ann.c0[i].value = model.stages[st].elems[el].C0[i].value

      X_raw = [pm_sim['T'],
        model.stages[st].elems[el].ann.F0.value,
        model.stages[st].elems[el].ann.p0.value * 1e5,
        model.stages[st].elems[el].ann.pp.value * 1e5] + \
        [model.stages[st].elems[el].ann.c0[i].value for i in range(n_solutes)]
      for i in pm['ann_input_idx']:
        model.stages[st].elems[el].ann.X_scaled[i].value = (X_raw[i] - pm['x_mean'][i]) / pm['x_scale'][i]

      x = np.array([model.stages[st].elems[el].ann.X_scaled[i].value for i in pm['ann_input_idx']])
      for l in pm['hidden_layers']:
        W = pm['ann_weights'][l]
        b_bias = pm['ann_biases'][l]
        z = W @ x + b_bias
        h = np.tanh(z)
        for n in pm['neurons'][l]:
          model.stages[st].elems[el].ann.H[l,n].value = h[n]
        x = h

      last = len(pm['ann_weights']) - 1
      W = pm['ann_weights'][last]
      b_bias = pm['ann_biases'][last]
      y_lin = W @ x + b_bias
      for n in pm['ann_output_idx']:
        model.stages[st].elems[el].ann.Y_lin[n].value = y_lin[n]

      for n in pm['ann_output_idx']:
        model.stages[st].elems[el].ann.Y_scaled[n].value = model.stages[st].elems[el].ann.Y_lin[n].value * pm['y_scale'][n] + pm['y_mean'][n]

      for i in range(n_solutes):
        model.stages[st].elems[el].ann.cp[i].value = np.exp(model.stages[st].elems[el].ann.Y_scaled[i].value)
      model.stages[st].elems[el].ann.Fp.value = model.stages[st].elems[el].ann.Y_scaled[6].value
      model.stages[st].elems[el].ann.pr.value = model.stages[st].elems[el].ann.Y_scaled[7].value / 1e5
      model.stages[st].elems[el].ann.Fr.value = model.stages[st].elems[el].ann.F0.value - model.stages[st].elems[el].ann.Fp.value
      Fr_val = max(model.stages[st].elems[el].ann.Fr.value, 1e-12)
      for i in range(n_solutes):
        model.stages[st].elems[el].ann.cr[i].value = (
          model.stages[st].elems[el].ann.F0.value * model.stages[st].elems[el].ann.c0[i].value
          - model.stages[st].elems[el].ann.Fp.value * model.stages[st].elems[el].ann.cp[i].value
        ) / Fr_val

      model.stages[st].elems[el].Fr.value = model.stages[st].elems[el].ann.Fr.value
      model.stages[st].elems[el].Fp.value = model.stages[st].elems[el].ann.Fp.value
      model.stages[st].elems[el].pr.value = model.stages[st].elems[el].ann.pr.value
      
      for i in range(n_solutes):
        model.stages[st].elems[el].Cr[i].value = model.stages[st].elems[el].ann.cr[i].value
        model.stages[st].elems[el].Cp[i].value = model.stages[st].elems[el].ann.cp[i].value

  return model, status


def sim_validation(input_file,pm_sim):
  df = pd.read_csv(f"{input_file}.csv")

  length = len(df.index)
  SIM_WATER_REC = np.zeros(length)
  SIM_DIVALENT_REC = np.zeros(length)
  SIM_SEP_FACTOR = np.zeros(length)
  SIM_MOL_POWER = np.zeros(length)
  VAL = []

  for i in range(length):
    process_parameters = {}
    pp_arr = np.array(ast.literal_eval(df['PERMEATE_PRESSURES'].iloc[i]))
    dn_arr = np.array(ast.literal_eval(df['DILUTION_FRACTIONS'].iloc[i]))
    f_arr = np.array(ast.literal_eval(df['FEED_FRACTIONS'].iloc[i]))
    ln_arr =  np.array(ast.literal_eval(df['LAMBDA_NORM'].iloc[i]))
    pn_arr = np.array(ast.literal_eval(df['PI_NORM'].iloc[i]))

    process_parameters['p_feed'] = df['FEED_PRESSURE'].iloc[i]*1e5
    process_parameters['pp_list'] = np.where(pp_arr == 0, 0.0, pp_arr)*1e5
    process_parameters['dNorm_vector'] = np.where(dn_arr == 0, 0.0, dn_arr)
    process_parameters['fNorm_vector'] = np.where(f_arr == 0, 0.0, f_arr)
    process_parameters['LambdaNorm_mx'] = np.where(ln_arr == 0, 0.0, ln_arr)
    process_parameters['PiNorm_mx'] = np.where(pn_arr == 0, 0.0, pn_arr)

    nf_system = nf_simulation.NFcascade(pm_sim, process_parameters)
    try:
      nf_system.simulate(epsilon=1.0)
      desc, sol_dict = nf_system.compute_descriptors()

      SIM_WATER_REC[i] = desc['water_recovery']
      SIM_DIVALENT_REC[i] = desc['recovery']
      SIM_SEP_FACTOR[i] = desc['separation_factor']
      SIM_MOL_POWER[i] = desc['molar_power']
      VAL.append(1.0)
    except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
      print(f'Validation of solution point number {i} failed due to math error: {e}')
      VAL.append(0.0)
    except Exception as e:
      print(f'Validation of solution point number {i} failed due to unexpected error: {e}')
      VAL.append(0.0)

  df['SIM_WATER_REC'] = SIM_WATER_REC
  df['SIM_DIVALENT_REC'] = SIM_DIVALENT_REC
  df['SIM_SEP_FACTOR'] = SIM_SEP_FACTOR
  df['SIM_MOL_POWER'] = SIM_MOL_POWER / 1e5

  df.to_csv(f"{input_file}_validation.csv")
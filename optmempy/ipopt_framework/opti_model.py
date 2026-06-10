import numpy as np
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints

def NFSystem_model(constraints, objective, pm):
  model = ConcreteModel()

  model_x = pm['model_x']
  relaxation = pm['relax']

  if pm['transport_model'] == 'sdec':
    from optmempy.ipopt_framework import optimization_sdec as opt
  else:
    print('Unknown model')
    return

  opt.initialize_model(pm)

  model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 1.0)
  model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
  model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)

  model.dilution_ratio = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)

  model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
  model.p_dil_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
  model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)

  model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0)
  model.mol_power = Var(within=NonNegativeReals, bounds = (0,100000), initialize = 1500.0)

  model.retentate_flow_rate = Var(within=NonNegativeReals, bounds=(1e-8,pm['F_lim']), initialize = pm['F_feed'])
  model.permeate_flow_rate = Var(within=NonNegativeReals, bounds=(1e-8,pm['F_lim']), initialize = pm['F_feed'])
  model.retentate_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
  model.retentate_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cr'][i], initialize = pm['c_feed_dict'])
  model.permeate_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['Cp'][i], initialize = pm['c_feed_dict'])

  model.Lambda = Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
  model.Pi = Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = 0.0)
  model.d = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_dil_feed']), initialize = 0.0)
  model.f = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'] / pm['nst'])

  if relaxation:
    model.LambdaNorm = Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.PiNorm = Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.dNorm = Var(pm['stages'], within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.fNorm = Var(pm['stages'], within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
  else:
    model.LambdaNorm = Var(pm['stages'], pm['stages'], domain=Binary, initialize=0)
    model.PiNorm     = Var(pm['stages'], pm['stages'], domain=Binary, initialize=0)
    model.dNorm      = Var(pm['stages'], domain=Binary, initialize=0)
    model.fNorm      = Var(pm['stages'], domain=Binary, initialize=0)


  model.Psi_Lambda =  Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,30000), initialize = 1500.0)
  model.Psi_Pi = Var(pm['stages'], pm['stages'], within=NonNegativeReals, bounds=(0,30000), initialize = 1500.0)
  model.Psi_d = Var(pm['stages'],within=NonNegativeReals, bounds=(0,30000), initialize = 1500.0)
  model.Psi_f = Var(pm['stages'],within=NonNegativeReals, bounds=(0,30000), initialize = 1500.0)

  model.stages = Block(pm['stages'], rule=lambda b: opt.stage_rule(b,pm))

  # PERFORMANCE METRICS
  def recovery_rule(model):
      return (model.retentate_flow_rate * (model.retentate_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][1])) == model.recovery
  model.recovery_constr = Constraint(rule=recovery_rule)

  def water_recovery_rule(model):
      return 1 - (model.retentate_flow_rate/(pm['F_feed'] + sum(model.d[i] * pm['F_dil_feed'] for i in pm['stages']))) == model.water_recovery
  model.water_recovery_constr = Constraint(rule=water_recovery_rule)

  def separation_factor_rule(model):
      return model.separation_factor == ( ( (model.retentate_concentration[1])/(model.retentate_concentration[0]+model.retentate_concentration[2]+model.retentate_concentration[3]) ) / ( (pm['c_feed_dict'][1])/(pm['c_feed_dict'][0]+pm['c_feed_dict'][2]+pm['c_feed_dict'][3]) ) )
  model.separation_factor_constr = Constraint(rule=separation_factor_rule)

  # SPLITTING AND MIXING CONSTRAINTS
  def Lambda_rule(model,st):
      return sum(model.Lambda[st,i] for i in range(pm['nst'])) <= model.stages[st].Fr
  model.Lambda_constraint = Constraint(pm['stages'],rule=Lambda_rule)

  def Pi_rule(model,st):
      return sum(model.Pi[st,i] for i in range(pm['nst'])) <= model.stages[st].Fp
  model.Pi_constraint = Constraint(pm['stages'],rule=Pi_rule)

  def d_rule(model):
      return sum(model.d[i] for i in pm['stages']) <= pm['F_dil_feed']
  model.d_constraint = Constraint(rule=d_rule)

  def f_rule(model):
      return sum(model.f[i] for i in pm['stages']) == pm['F_feed']
  model.f_constraint = Constraint(rule=f_rule)

  # RATIONALIZATIONS
  def SelfLambda_rule(model,st):
      return model.Lambda[st,st] == 0.0
  model.SelfLambda_constraint = Constraint(pm['stages'],rule=SelfLambda_rule)

  def SelfPi_rule(model,st):
      return model.Pi[st,st] == 0.0
  model.SelfPi_constraint = Constraint(pm['stages'],rule=SelfPi_rule)

  # NORMALIZATIONS
  def LambdaNorm_rule(model,origin,target):
      return model.LambdaNorm[origin, target] * model.stages[origin].Fr == model.Lambda[origin, target]
  model.LambdaNorm_constraint = Constraint(pm['stages'], pm['stages'], rule=LambdaNorm_rule)

  def PiNorm_rule(model,origin,target):
      return model.PiNorm[origin, target] * model.stages[origin].Fp == model.Pi[origin, target]
  model.PiNorm_constraint = Constraint(pm['stages'], pm['stages'], rule=PiNorm_rule)

  def dNorm_rule(model,target):
      return model.dNorm[target] * pm['F_dil_feed'] == model.d[target]
  model.dNorm_constraint = Constraint(pm['stages'], rule = dNorm_rule)

  def fNorm_rule(model,target):
      return model.fNorm[target] * pm['F_feed'] == model.f[target]
  model.fNorm_constraint = Constraint(pm['stages'], rule = fNorm_rule)

  # OVERALL MASS BALANCE
  def omb_stage_mixing(model,st):
      return model.stages[st].F0 == sum(model.Lambda[i,st] for i in pm['stages']) + sum(model.Pi[i,st] for i in pm['stages'])  + model.d[st] + model.f[st]
  model.omb_stage_mixing = Constraint(pm['stages'],rule=omb_stage_mixing)

  def omb_retentate_mixing(model):
      return model.retentate_flow_rate == sum((model.stages[i].Fr - sum(model.Lambda[i,j] for j in pm['stages'])) for i in pm['stages'])
  model.omb_retentate_mixing = Constraint(rule=omb_retentate_mixing)

  def omb_permeate_mixing(model):
      return model.permeate_flow_rate == sum((model.stages[i].Fp - sum(model.Pi[i,j] for j in pm['stages'])) for i in pm['stages'])
  model.omb_permeate_mixing = Constraint(rule=omb_permeate_mixing)

  # COMPONENT MASS BALANCE
  def cmb_stage_mixing(model,st,s):
      return model.stages[st].F0 * model.stages[st].C0[s] == sum(model.Lambda[i,st] * model.stages[i].Cr[s] for i in pm['stages']) + sum(model.Pi[i,st] * model.stages[i].Cp[s] for i in pm['stages'])  + model.f[st] * pm['c_feed_dict'][s]
  model.cmb_stage_mixing = Constraint(pm['stages'],pm['solutes'],rule=cmb_stage_mixing)

  def cmb_retentate_mixing(model,s):
      return model.retentate_flow_rate * model.retentate_concentration[s] == sum((model.stages[i].Fr * model.stages[i].Cr[s] - sum(model.Lambda[i,j] * model.stages[i].Cr[s] for j in pm['stages'])) for i in pm['stages'])
  model.cmb_retentate_mixing = Constraint(pm['solutes'],rule=cmb_retentate_mixing)

  def cmb_permeate_mixing(model,s):
      return model.permeate_flow_rate * model.permeate_concentration[s] == sum((model.stages[i].Fp * model.stages[i].Cp[s] - sum(model.Pi[i,j] * model.stages[i].Cp[s] for j in pm['stages'])) for i in pm['stages'])
  model.cmb_permeate_mixing = Constraint(pm['solutes'],rule=cmb_permeate_mixing)

  # STAGE CUT RULE
  def stage_cut_rule(model,st):
      return model.stages[st].Fp <= 0.85 * model.stages[st].F0
  model.stage_cut_constraint = Constraint(pm['stages'],rule=stage_cut_rule)

  # CONCENTRATION FACTOR RULE
  def concentration_factor_rule(model):
      return ( model.retentate_concentration[0] + model.retentate_concentration[1] ) / ( pm['c_feed_dict'][0] + pm['c_feed_dict'][1] ) <= pm['CF_max']
  model.concentration_factor_constraint = Constraint(rule=concentration_factor_rule)

  # CASCADE FLOW RULE
  if pm['CF_max'] < 10:
    def cascade_flow_rule(model):
        return sum(model.stages[st].F0 for st in model.stages) <= 3 * pm['F_feed']
    model.cascade_flow_constraint = Constraint(rule=cascade_flow_rule)

  # PRESSURE BALANCE
  def stage_pressure_rule(model,st):
      return model.stages[st].p0 == model.p_feed
  model.stage_pressure_constraint = Constraint(pm['stages'],rule=stage_pressure_rule)

  def retentate_pressure_rule(model):
      return model.retentate_pressure == model.p_feed
  model.retentate_pressure_constraint = Constraint(rule=retentate_pressure_rule)

  # PRESSURE EXCHANGER AND POWER CONSTRAINTS
  def dilution_ratio_rule(model):
      return model.dilution_ratio == sum(model.d[i] for i in pm['stages']) / (sum(model.d[i] for i in pm['stages']) + pm['F_feed'])
  model.dilution_ratio_constraint = Constraint(rule=dilution_ratio_rule)

  if model_x == False:
    def pex_rule1(model):
        return model.p_feed == model.p_pump
    def pex_rule2(model):
        return model.p_feed == model.p_dil_pump
  else:
    def pex_rule1(model):
        return model.p_feed * pm['F_feed'] == pm['pex_eff'] * model.retentate_flow_rate * model.retentate_pressure * (1-model.dilution_ratio) + pm['F_feed'] * model.p_pump
    def pex_rule2(model):
        return model.p_feed * sum(model.d[i] for i in pm['stages']) == pm['pex_eff'] * model.retentate_flow_rate * model.retentate_pressure * model.dilution_ratio + sum(model.d[i] for i in pm['stages']) * model.p_dil_pump
  model.pex_constraint1 = Constraint(rule=pex_rule1)
  model.pex_constraint2 = Constraint(rule=pex_rule2)

  def dilution_power_rule(model,st):
      return model.Psi_d[st] == (1/pm['pump_eff']) * model.d[st] * model.p_dil_pump
  model.dilution_power_constr = Constraint(pm['stages'],rule=dilution_power_rule)

  def retentate_power_rule(model,origin,target):
      return model.Psi_Lambda[origin,target] == (1/pm['pump_eff']) * (model.stages[target].p0 - model.stages[origin].pr) * model.Lambda[origin,target]
  model.retentate_power_constr = Constraint(pm['stages'],pm['stages'],rule=retentate_power_rule)

  def permeate_power_rule(model,origin,target):
      return model.Psi_Pi[origin,target] == (1/pm['pump_eff']) * (model.stages[target].p0 - model.stages[origin].pp) * model.Pi[origin,target]
  model.permeate_power_constr = Constraint(pm['stages'],pm['stages'],rule=permeate_power_rule)

  def power_rule(model):
      return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump + sum(model.Psi_Lambda[i,j] for i in pm['stages'] for j in pm['stages']) + sum(model.Psi_Pi[i,j] for i in pm['stages']  for j in pm['stages']) + sum(model.Psi_d[i] for i in pm['stages'])
  model.power_constr = Constraint(rule=power_rule)

  def molar_power_rule(model):
      return model.mol_power == model.power / model.recovery
  model.molar_power_constr = Constraint(rule=molar_power_rule)

  # MULTIPLE OBJECTIVE CONSTRAINTS
  if 'recovery' in constraints:
    def mo_rec_rule(model):
        return model.recovery >= constraints['recovery']
    model.mo_rec = Constraint(rule=mo_rec_rule)
  
  if 'separation_factor' in constraints:
    def mo_sf_rule(model):
        return model.separation_factor >= constraints['separation_factor']
    model.mo_sf = Constraint(rule=mo_sf_rule)

  if 'molar_power' in constraints:
    def mo_sp_rule(model):
        return model.mol_power <= constraints['molar_power']
    model.mo_sp = Constraint(rule=mo_sp_rule)

  # OBJECTIVE FUNCTION
  if objective =='separation_factor':
    model.obj = Objective(expr=(-( model.separation_factor )))
  elif objective == 'recovery':
    model.obj = Objective(expr=(-( model.recovery )))
  elif objective =='molar_power':
    model.obj = Objective(expr=(model.mol_power))
  else:
    print('Unknown objective function. Falling back to default: separation factor')
    model.obj = Objective(expr=(-( model.separation_factor )))
    return

  return model


def optimize(model,solver = 'ipopt'):
  try:
    if solver == 'ipopt':
      with SolverFactory('ipopt') as opt:
        """
        opt.options.update({"nlp_scaling_method": "user-scaling",
                            "tol": 1e-6,
                            "acceptable_tol": 1e-6,
                            "acceptable_constr_viol_tol": 1e-6,
                            "acceptable_dual_inf_tol": 1e-6,
                            "acceptable_obj_change_tol": 1e-7,
                            "max_iter": 5000})
        """                
        results = opt.solve(model, tee=True)
    elif solver == 'baron':
      with SolverFactory('baron') as opt:
        results = opt.solve(model, tee=True, options={"MaxTime": -1})
    elif solver == 'couenne':
      print('COUENNE NOT INSTALLED')
      return model, None, 0
    else:
      print('Solver not recognized.')
      return model, None, 0

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
      optimal = 1
    else:
      optimal = 0
  except ValueError:
    results = {}
    optimal = 0

  return model, results, optimal


def extract_results(model,n_stages):
  results = {}
  try:
    results['p_feed'] = value(model.p_feed)
  except:
    pass
  try:
    results['p_pump'] = value(model.p_pump)
    results['p_feed'] = value(model.p_feed)
  except:
    pass

  p_perm = np.zeros((n_stages))
  for i in range(n_stages):
    p_perm[i] = value(model.stages[i].pp)
  results['p_perm'] = p_perm

  try:
    lamda_mx = np.zeros((n_stages,n_stages))
    for i in range(n_stages):
      for j in range(n_stages):
        lamda_mx[i,j] = value(model.Lambda[i,j])
    results['lambda_mx'] = lamda_mx
  except:
    pass

  try:
    pi_mx = np.zeros((n_stages,n_stages))
    for i in range(n_stages):
      for j in range(n_stages):
        pi_mx[i,j] = value(model.Pi[i,j])
    results['pi_mx'] = pi_mx
  except:
    pass

  try:
    dilutions = np.zeros((n_stages))
    for i in range(n_stages):
      dilutions[i] = value(model.d[i])
    results['d'] = dilutions
  except:
    pass

  results['retentate_separation_factor'] = value(model.separation_factor)
  results['molar_power'] = value(model.mol_power)
  results['recovery'] = value(model.recovery)

  return results


def log_opti(model):
  import logging

  logger = logging.getLogger()

  logger.setLevel(logging.INFO)
  log_infeasible_constraints(model)

  pyomo.util.infeasible.log_close_to_bounds(model)
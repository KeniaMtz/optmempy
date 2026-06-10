import numpy as np
from pyomo.environ import *
from pyomo.environ import exp, tanh
from pathlib import Path
import optmempy.nf_cascade_simulation as nfcs

def initialize_model(pm):
  BASE_DIR = Path(nfcs.__file__).resolve().parent
  data = np.load(BASE_DIR / "ann" / "ann_export.npz")

  pm['ann_weights'] = [data[k] for k in sorted(data.keys()) if k.startswith("W")]
  pm['ann_biases'] = [data[k] for k in sorted(data.keys()) if k.startswith("b") and not k.endswith("mean") and not k.endswith("scale")]

  pm['x_mean'] = data["x_mean"]
  pm['x_scale'] = data["x_scale"]
  pm['y_mean'] = data["y_mean"]
  pm['y_scale'] = data["y_scale"]

  pm['hidden_layers'] = range(len(pm['ann_weights']) - 1)
  pm['neurons'] = {l: range(pm['ann_weights'][l].shape[0]) for l in pm['hidden_layers']}

  pm['layer_indices'] = range(len(pm['ann_weights']))
  pm['neuron_indices'] = {layer: range(pm['ann_weights'][layer].shape[0]) for layer in pm['layer_indices']}
  pm['ann_input_idx'] = range(10)
  pm['ann_output_idx'] = range(8)

  pm['stages'] = range(pm['nst'])
  pm['elems'] = range(pm['ne'])
  pm['red_elems'] = range(pm['ne']-1)
  pm['solutes'] = range(pm['ns'])

  C_bound = {}
  for i in pm['solutes']:
    C_bound[i] = (0,1000)
  pm['bounds'] = {'C': C_bound, 'Cr': C_bound, 'Cp' : C_bound}


def ann_rule(b, pm):
  b.F0 = Var(within=NonNegativeReals, bounds=(1e-8, pm['F_lim']))
  b.p0 = Var(within=NonNegativeReals, bounds=(0, pm['dp_max']))
  b.pp = Var(within=NonNegativeReals, bounds=(0, pm['dp_max']))
  b.c0 = Var(pm['solutes'], within=NonNegativeReals)

  b.T  = Param(initialize=pm['T'])

  b.X_scaled = Var(pm['ann_input_idx'], within=Reals)
  b.H = Var([(l, n) for l in pm['hidden_layers'] for n in pm['neurons'][l]], bounds=(-1.1, 1.1))
  b.Y_lin = Var(pm['ann_output_idx'], within=Reals)
  b.Y_scaled = Var(pm['ann_output_idx'], within=Reals)

  b.cp = Var(pm['solutes'], within=NonNegativeReals)
  b.Fp = Var(within=NonNegativeReals, bounds=(1e-8, pm['F_lim']))
  b.pr = Var(within=NonNegativeReals, bounds=(0, pm['dp_max']))

  b.Fr = Var(within=NonNegativeReals, bounds=(1e-8, pm['F_lim']))
  b.cr = Var(pm['solutes'], within=NonNegativeReals)

  def input_scaling_rule(b, i):
    if i == 0: val = b.T
    elif i == 1: val = b.F0
    elif i == 2: val = b.p0 * 1e5
    elif i == 3: val = b.pp * 1e5
    else: val = b.c0[i-4]
    return b.X_scaled[i] == (val - pm['x_mean'][i]) / pm['x_scale'][i]
  b.input_scaling = Constraint(pm['ann_input_idx'], rule=input_scaling_rule)

  def hidden_rule(b, l, n):
    W = pm['ann_weights'][l]
    bias = pm['ann_biases'][l]
    expr = sum(W[n,i] * (b.X_scaled[i] if l == 0 else b.H[l-1,i]) for i in range(W.shape[1]))
    return b.H[l,n] == tanh(expr + bias[n])
  b.hidden = Constraint([(l,n) for l in pm['hidden_layers'] for n in pm['neurons'][l]], rule=hidden_rule)

  last = len(pm['ann_weights']) - 1
  def output_lin_rule(b, n):
    W = pm['ann_weights'][last]
    bias = pm['ann_biases'][last]
    expr = sum(W[n,i] * b.H[last-1,i] for i in range(W.shape[1]))
    return b.Y_lin[n] == expr + bias[n]
  b.output_lin = Constraint(pm['ann_output_idx'], rule=output_lin_rule)

  def output_scaling_rule(b, n):
    return b.Y_scaled[n] == b.Y_lin[n] * pm['y_scale'][n] + pm['y_mean'][n]
  b.output_scaling = Constraint(pm['ann_output_idx'], rule=output_scaling_rule)

  def cp_rule(b, s):
    return b.cp[s] == exp(b.Y_scaled[s])
  b.cp_def = Constraint(pm['solutes'], rule=cp_rule)
  # Rest of the outputs
  b.Fp_def = Constraint(expr=b.Fp == b.Y_scaled[6])
  b.pr_def = Constraint(expr=b.pr == b.Y_scaled[7] / 1e5)

  b.mass_bal = Constraint(expr=b.F0 == b.Fr + b.Fp)
  def solute_balance_rule(b, s):
    return b.F0*b.c0[s] == b.Fr*b.cr[s] + b.Fp*b.cp[s]
  b.solute_bal = Constraint(pm['solutes'], rule=solute_balance_rule)


def element_rule(b,pm):
  b.F0 = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = pm['F_feed'])
  b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
  b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
  b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])

  b.Fr = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = pm['F_feed'])
  b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
  b.Fp = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = 1e-8)
  b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
  b.pr = Var(within=NonNegativeReals, bounds=(0,pm['dp_max']))
  
  b.ann = Block(rule=lambda b: ann_rule(b, pm))

  b.link_F0 = Constraint(expr=b.ann.F0 == b.F0)
  b.link_p0 = Constraint(expr=b.ann.p0 == b.p0)
  b.link_pp = Constraint(expr=b.ann.pp == b.pp)
  def link_C0_rule(b, s):
    return b.ann.c0[s] == b.C0[s]
  b.link_C0 = Constraint(pm['solutes'], rule=link_C0_rule)

  b.link_Fr = Constraint(expr=b.Fr == b.ann.Fr)
  b.link_Fp = Constraint(expr=b.Fp == b.ann.Fp)
  b.link_pr = Constraint(expr=b.pr == b.ann.pr)

  def link_Cr_rule(b, s):
    return b.Cr[s] == b.ann.cr[s]
  b.link_Cr = Constraint(pm['solutes'], rule=link_Cr_rule)

  def link_Cp_rule(b, s):
    return b.Cp[s] == b.ann.cp[s]
  b.link_Cp = Constraint(pm['solutes'], rule=link_Cp_rule)


def stage_rule(b,pm):
  b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
  b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
  b.pr = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
  
  b.F0 = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = pm['F_feed'])
  b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

  b.Fr = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = pm['F_feed'])
  b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
  
  b.Fp = Var(within=NonNegativeReals, bounds = (1e-8,pm['F_lim']), initialize = 1e-8)
  b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

  b.elems = Block(pm['elems'],rule=lambda b: element_rule(b,pm))
  
  def intermediary_retentate_rule(b,n):
    return b.elems[n].Fr == b.elems[n+1].F0

  def intermediary_retentate_concentration_rule(b,n,s):
    return b.elems[n].Cr[s] == b.elems[n+1].C0[s]
  
  def feed_rule(b):
    return b.F0 == b.elems[0].F0
  
  def feed_concentration_rule(b,s):
    return b.C0[s] == b.elems[0].C0[s]

  def retentate_rule(b):
    return b.Fr == b.elems[pm['ne']-1].Fr

  def retentate_concentration_rule(b,s):
    return b.Cr[s] == b.elems[pm['ne']-1].Cr[s]
  
  def permeate_rule(b):
    return b.Fp == sum(b.elems[i].Fp for i in pm['elems'])
  
  def permeate_concentration_rule(b,s):
    return b.Fp * b.Cp[s] == sum(b.elems[i].Fp * b.elems[i].Cp[s] for i in pm['elems'])
  
  def intermediary_pressure_rule(b,n):
    return b.elems[n+1].p0 == b.elems[n].pr

  def feed_pressure_rule(b):
    return b.p0 == b.elems[0].p0

  def retentate_pressure_rule(b):
    #return b.pr == b.elems[pm['ne']-1].p0
    return b.pr == b.elems[pm['ne']-1].pr
  
  def permeate_pressure_rule(b,n):
    return b.pp == b.elems[n].pp
  
  b.intermediary_retentate = Constraint(pm['red_elems'], rule = intermediary_retentate_rule)
  b.intermediary_retentate_concentration = Constraint(pm['red_elems'], pm['solutes'], rule = intermediary_retentate_concentration_rule)
  b.feed = Constraint(rule = feed_rule)
  b.feed_concentration = Constraint(pm['solutes'], rule = feed_concentration_rule)
  b.retentate = Constraint(rule = retentate_rule)
  b.retentate_concentration = Constraint(pm['solutes'], rule = retentate_concentration_rule)
  b.permeate = Constraint(rule = permeate_rule)
  b.permeate_concentration = Constraint(pm['solutes'], rule = permeate_concentration_rule)

  b.feed_pressure = Constraint(rule = feed_pressure_rule)
  b.intermediary_pressure = Constraint(pm['red_elems'], rule = intermediary_pressure_rule)
  b.retentate_pressure = Constraint(rule = retentate_pressure_rule)
  b.permeate_pressure = Constraint(pm['elems'],rule = permeate_pressure_rule)
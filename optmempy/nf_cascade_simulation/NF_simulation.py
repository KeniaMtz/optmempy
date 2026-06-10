import numpy as np
from optmempy.nf_cascade_simulation import ann_model as mdl
from copy import deepcopy

def within_percent_tolerance(arr: np.ndarray, ref: np.ndarray, minimum_magnitude: float, epsilon: float) -> bool:
  """
  Checks if all elements of 'arr' are within epsilon percent tolerance of 'ref'.
  Both arrays must have the same shape.

  - For ref[i, j] != 0: relative error <= epsilon%
  - For ref[i, j] == 0: absolute error <= epsilon% of mean(abs(ref))
  """
  if arr.shape != ref.shape:
    raise ValueError("arr and ref must have the same shape.")

  with np.errstate(divide='ignore', invalid='ignore'):
    diff = np.abs(arr - ref)
    mean_magnitude = max(np.mean(np.abs(ref)), minimum_magnitude)
    abs_tol = (epsilon / 100.0) * mean_magnitude

    rel_mask = (ref != 0)
    rel_error_ok = np.all((diff[rel_mask] / np.abs(ref[rel_mask])) <= (epsilon / 100.0))

    abs_mask = ~rel_mask
    abs_error_ok = np.all(diff[abs_mask] <= abs_tol)

  return rel_error_ok and abs_error_ok


class Stage:
  def __init__(self, stg_index, p_feed, pm):
    self.index = stg_index
    self.pm = pm
    self.simulation = {
      'Fr': 0.0,
      'cr': [0.0] * pm['ns'],
      'pr': p_feed,
      'Fp': 0.0,
      'cp': [0.0] * pm['ns'],
      'elements': None,
      'p0': p_feed,
      'c0': pm['c_feed'],
      'F0': 0.0,
      'pp': 0.0,
    }

  def mix_inlet(self, F_list, c_list, F_dil=0.0):
    """
    Compute stage inlet flow F0 and inlet concentration c0
    by mixing incoming streams and optional dilution.

    Parameters
    ----------
    F_list : list of float
      F_list = [F₁, F₂, ..., Fₙ]
      Volumetric flowrates of incoming streams [m3/h]
    c_list : list of array-like
      c_list = [c₁, c₂, ..., cₙ] where each cⱼ is a vector of length ns
      Concentration vectors of incoming streams [mol/m3]
    F_dil : float, optional
      Dilution flowrate (pure water), by default 0.0
    """
    ns = self.pm['ns']
    assert len(F_list) == len(c_list), f"Mismatch: F_list ({len(F_list)}) != c_list ({len(c_list)})"

    self.F0 = np.sum(F_list) + F_dil
    if self.F0 == 0:
      self.c0 = [0.0] * ns
      return
    if self.F0 < 0:
      raise ValueError(f"Physical Error: Total inlet flow F0 is negative ({self.F0})")

    c_array = np.asarray(c_list, dtype=float)
    f_array = np.asarray(F_list, dtype=float)
    molar_flow = np.dot(f_array, c_array)
    inlet_conc = molar_flow / self.F0
    self.c0 = inlet_conc.tolist()

  def simulate(self, F_list, c_list, p0, pp, F_dil=0.0):
    self.mix_inlet(F_list, c_list, F_dil)
    ns = self.pm['ns']
    ne = self.pm['ne']
    T = self.pm['T']

    solutions = []

    # First element
    args = [T, self.F0, p0, pp, self.c0]
    sol = mdl.run_ann(args)
    solutions.append(sol)

    # Remaining elements
    for _ in range(ne - 1):
      args = [
        T, solutions[-1]['Fr'], solutions[-1]['pr'], pp, solutions[-1]['cr']
      ]
      sol = mdl.run_ann(args)
      solutions.append(sol)

    Fp_total = sum(e['Fp'] for e in solutions)
    if Fp_total > 0:
      cp_final = (np.sum([e['Fp'] * np.array(e['cp']) for e in solutions], axis=0) / Fp_total)
    else:
      cp_final = np.zeros(ns)

    self.simulation = {
      'Fr': solutions[-1]['Fr'],
      'cr': solutions[-1]['cr'],
      'pr': solutions[-1]['pr'],
      'Fp': Fp_total,
      'cp': cp_final.tolist(),
      'elements': solutions,
      'p0': p0,
      'c0': self.c0,
      'F0': self.F0,
      'pp': pp,
    }


class NFcascade:
  def __init__(self, pm, process_pm):
    self.pm = pm
    self.ns = pm['ns']
    self.model_x = pm['model_x']
    self.pex_eff = pm['pex_eff']
    self.pump_eff = pm['pump_eff']
    self.n_stages = pm['nst']
    self.F_feed = pm['F_feed']
    self.c_feed = pm['c_feed']
    self.p_feed = process_pm['p_feed']
    self.pp_list = process_pm['pp_list']
    self.LambdaNorm_mx = process_pm['LambdaNorm_mx']
    self.PiNorm_mx = process_pm['PiNorm_mx']
    self.dNorm_vector = process_pm['dNorm_vector']
    self.fNorm_vector = process_pm['fNorm_vector']
    # Stage-local initialization for iteration 0
    self.stages = [Stage(i, self.p_feed, pm) for i in range(self.n_stages)]

  def simulate(self, epsilon=1.0):
    ns = self.ns
    F_dil_feed = self.pm['F_dil_feed']
    self.F_dil_list = [self.dNorm_vector[i] * F_dil_feed for i in range(self.n_stages)]
    F_feed_split = [self.fNorm_vector[i] * self.F_feed for i in range(self.n_stages)]

    self.Lambda_mx = np.zeros_like(self.LambdaNorm_mx)
    self.Pi_mx = np.zeros_like(self.PiNorm_mx)

    counter = 0

    while True:
      self.Lambda_mx_new = np.zeros_like(self.Lambda_mx)
      self.Pi_mx_new = np.zeros_like(self.Pi_mx)

      for i in range(self.n_stages):
        F_inputs, c_inputs = [], []

        F_inputs.append(F_feed_split[i])
        c_inputs.append(self.c_feed)

        for j in range(self.n_stages):
          if self.Lambda_mx[j, i] > 0:
            F_inputs.append(self.Lambda_mx[j, i])
            c_inputs.append(self.stages[j].simulation['cr'])
        
        for j in range(self.n_stages):
          if self.Pi_mx[j, i] > 0:
            F_inputs.append(self.Pi_mx[j, i])
            c_inputs.append(self.stages[j].simulation['cp'])

        self.stages[i].simulate(
          F_list=F_inputs,
          c_list=c_inputs,
          p0=self.p_feed,
          pp=self.pp_list[i],
          F_dil=self.F_dil_list[i]
        )

        if np.isnan(self.stages[i].simulation['Fr']):
          self.stages[i].simulation['Fr'] = 0.0
        if np.isnan(self.stages[i].simulation['Fp']):
          self.stages[i].simulation['Fp'] = 0.0
        for solute in range(ns):
          if np.isnan(self.stages[i].simulation['cr'][solute]):
            self.stages[i].simulation['cr'][solute] = 0.0
          if np.isnan(self.stages[i].simulation['cp'][solute]):
            self.stages[i].simulation['cp'][solute] = 0.0

      for i in range(self.n_stages):
        for j in range(self.n_stages):
          self.Lambda_mx_new[j, i] = self.stages[j].simulation['Fr'] * self.LambdaNorm_mx[j, i]
          self.Pi_mx_new[j, i] = self.stages[j].simulation['Fp'] * self.PiNorm_mx[j, i]

      self.F_retentate = sum((self.stages[i].simulation['Fr'] - sum(self.Lambda_mx_new[i,j] for j in range(self.n_stages))) for i in range(self.n_stages))
      self.F_permeate = sum((self.stages[i].simulation['Fp'] - sum(self.Pi_mx_new[i,j] for j in range(self.n_stages))) for i in range(self.n_stages))
      self.c_retentate = np.zeros(ns)
      self.c_permeate = np.zeros(ns)
      for k in range(ns):
        self.c_retentate[k] = sum((self.stages[i].simulation['Fr'] * self.stages[i].simulation['cr'][k] - sum(self.Lambda_mx_new[i,j] * self.stages[i].simulation['cr'][k] for j in range(self.n_stages))) for i in range(self.n_stages)) / self.F_retentate
        self.c_permeate[k] = sum((self.stages[i].simulation['Fp'] * self.stages[i].simulation['cp'][k] - sum(self.Pi_mx_new[i,j] * self.stages[i].simulation['cp'][k] for j in range(self.n_stages))) for i in range(self.n_stages)) / self.F_permeate

      if (
        within_percent_tolerance(self.Lambda_mx_new, self.Lambda_mx, self.F_feed, epsilon)
        and
        within_percent_tolerance(self.Pi_mx_new, self.Pi_mx, self.F_feed, epsilon)
        and counter > self.n_stages + 1
      ):
        print('Superstructure converged after {} iterations'.format(counter))
        break

      if counter >= 100:
        print('Superstructure did NOT converge after {} iterations'.format(counter))
        raise ValueError("Superstructure did not converge")

      self.Lambda_mx = deepcopy(self.Lambda_mx_new)
      self.Pi_mx = deepcopy(self.Pi_mx_new)
      counter += 1

  def compute_descriptors(self):
    # Magnesium recovery - η
    numerator = (self.F_retentate * (self.c_retentate[1]))
    denumerator = (self.F_feed * (self.c_feed[1]))
    recovery = numerator / denumerator

    # Water recovery - η_W
    water_recover = 1 - ( self.F_retentate / ( self.F_feed + np.sum(self.F_dil_list) ) )

    # Separation factor - β
    numerator_1 = ( self.c_retentate[1] )
    denumerator_1 = ( self.c_retentate[0] + self.c_retentate[2] + self.c_retentate[3] )
    numerator_2 = ( self.c_feed[1] )
    denumerator_2 = ( self.c_feed[0] + self.c_feed[2] + self.c_feed[3] )
    sep_factor = ( numerator_1 / denumerator_1 ) / ( numerator_2 / denumerator_2 )

    # Power demand - Ψ
    dilution_ratio = sum(self.F_dil_list[i] for i in range(self.n_stages)) / (sum(self.F_dil_list[i] for i in range(self.n_stages)) + self.F_feed)
    # Pump pressure
    if self.model_x:
      p_pump = (self.F_feed * self.p_feed - self.pm['pex_eff'] * self.F_retentate * self.p_feed * (1-dilution_ratio)) / self.F_feed   # Pressure exchanger
      if sum(self.F_dil_list[i] for i in range(self.n_stages)) < 1e-5:
        p_dil_pump = self.p_feed
      else:
        p_dil_pump = (sum(self.F_dil_list[i] for i in range(self.n_stages)) * self.p_feed - self.pm['pex_eff'] * self.F_retentate * self.p_feed * dilution_ratio) / sum(self.F_dil_list[i] for i in range(self.n_stages))   # Dilution pump
    else:
      p_pump = self.p_feed
      p_dil_pump = self.p_feed
    # Dilution power - Ψ_dilution
    psi_dil = np.array([(1 / self.pump_eff) * self.F_dil_list[i] * p_dil_pump for i in range(self.n_stages)])
    # Retentate (Lambda) and permeate (Pi) recirculation power - Ψ_recycling
    psi_Lambda = np.zeros_like(self.Lambda_mx)
    psi_Pi = np.zeros_like(self.Pi_mx)
    for i in range(self.n_stages):
      for j in range(self.n_stages):
        if self.Lambda_mx[i, j] > 0:
          psi_Lambda[i, j] = (1 / self.pm['pump_eff']) * self.stages[i].simulation['Fr'] * self.Lambda_mx[i, j] * (self.stages[j].simulation['p0'] - self.stages[i].simulation['pr'])
        if self.Pi_mx[i, j] > 0:
          psi_Pi[i, j] = (1 / self.pm['pump_eff']) * self.stages[i].simulation['Fp'] * self.Pi_mx[i, j] * (self.stages[j].simulation['p0'] - self.stages[i].simulation['pp'])
    total_power = (
      (1 / self.pm['pump_eff']) * self.F_feed * p_pump
      + np.sum(psi_dil)
      + np.sum(psi_Lambda)
      + np.sum(psi_Pi)
    )
    molar_power = total_power / recovery

    descriptors = {
      "Lambda_mx": self.Lambda_mx,
      "Pi_mx": self.Pi_mx,
      "c_retentate": self.c_retentate,
      "c_permeate": self.c_permeate,
      "recovery" : recovery,
      "separation_factor" : sep_factor,
      "water_recovery" : water_recover,
      "p_pump": p_pump,
      "total_power": total_power,
      "molar_power" : molar_power,
      "Psi_d": psi_dil,
      "Psi_Lambda": psi_Lambda,
      "Psi_Pi": psi_Pi,
      "retentate_flow_rate": self.F_retentate,
      "permeate_flow_rate": self.F_permeate,
      "retentate_pressure": self.p_feed,
      "dilution_ratio": dilution_ratio,
      "p_dil_pump": p_dil_pump
    }

    return descriptors, self.stages
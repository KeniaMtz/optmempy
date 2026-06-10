import numpy as np
from optmempy.nf_cascade_simulation import NF_simulation as simu

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
}

process_parameters = {
  'p_feed'          :  22.4e5,  # Pa
  'pp_list'  : [3e5, 3e5, 3e5, 3e5, 0, 0], # Pa
  'LambdaNorm_mx'   : np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
  'PiNorm_mx'       : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
  'dNorm_vector': np.array([0, 0, 0, 0, 0, 0]),
  'fNorm_vector': np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
}



sim_cascade = simu.NFcascade(pm_sim, process_parameters)
sim_cascade.simulate()

desc, sol_dict = sim_cascade.compute_descriptors()

print(desc['recovery'],desc['separation_factor'],(desc['molar_power']/100000)*100000/(3600*1000))
import numpy as np
import maingopy
from optmempy.maingo_framework.utils import (
  scale_single_value,
  get_idx_map,
)

class ANNWrapper:
    def __init__(self, ann_path, ann_name):
        self.ann = maingopy.melonpy.FeedForwardNet()
        self.ann.load_model(str(ann_path), ann_name, maingopy.melonpy.XML)

    def predict(self, x_vector):
        return self.ann.calculate_prediction_reduced_space(x_vector)

# Objective
def separation_factor_objective(x_vector, ann_model):
    ann_y = ann_model.predict(x_vector)
    sep_scaled = ann_y[1]
    return -sep_scaled

def molar_power_objective(x_vector, ann_model):
    ann_y = ann_model.predict(x_vector)
    mpower_scaled = ann_y[2]
    return mpower_scaled

# Constraints
def divalent_recovery_constraint(x_vector, ann_model, R_min_scaled):
    ann_y = ann_model.predict(x_vector)
    recovery_scaled = ann_y[0]
    return R_min_scaled - recovery_scaled  # ≤ 0

def separation_factor_constraint(x_vector, ann_model, Beta_min_scaled):
    ann_y = ann_model.predict(x_vector)
    Beta_scaled = ann_y[1]
    return Beta_min_scaled - Beta_scaled

def pressure_constraint(x_vector_scaled, pm):
    ns = pm['nst']
    idx = get_idx_map(ns)
    const = []
    
    pf0, _ = idx['p_feed']
    p_feed_scaled = x_vector_scaled[pf0]
    pp0, pp_1 = idx['pp']
    pp_scaled = x_vector_scaled[pp0:pp_1]

    for pp_i in pp_scaled:
        const.append(pp_i - p_feed_scaled)
    return const

# Bounds
def get_bounds(pm, x_scaler, scale_idx):
    ns = pm['nst']
    lb_real = []
    ub_real = []

    # Bounds in the physical scale
    # p_feed
    lb_real.append(20)
    ub_real.append(pm['dp_max'])
    
    # pp_list
    lb_real += [0] * ns
    ub_real += [pm['dp_max']] * ns

    lb_real = np.array(lb_real)
    ub_real = np.array(ub_real)

    # Scaled bounds
    lb_scaled = lb_real.copy()
    ub_scaled = ub_real.copy()

    idx = get_idx_map(ns)
    pf0, _ = idx['p_feed']
    pp0, _ = idx['pp']

    if pf0 in scale_idx:
        lb_scaled[0] = scale_single_value(lb_real[0], pf0, x_scaler)
        ub_scaled[0] = scale_single_value(ub_real[0], pf0, x_scaler)

    for i in range(ns):
        ann_idx = pp0 + i
        vector_idx = 1 + i
        if ann_idx in scale_idx:
            lb_scaled[vector_idx] = scale_single_value(lb_real[vector_idx], ann_idx, x_scaler)
            ub_scaled[vector_idx] = scale_single_value(ub_real[vector_idx], ann_idx, x_scaler)

    return lb_scaled, ub_scaled

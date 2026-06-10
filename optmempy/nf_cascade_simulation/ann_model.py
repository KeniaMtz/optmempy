import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
npz_path = BASE_DIR / "ann" / "ann_export.npz"
data = np.load(npz_path)

x_mean = data["x_mean"]
x_scale = data["x_scale"]
y_mean = data["y_mean"]
y_scale = data["y_scale"]

Ws = [data[k] for k in sorted(data.keys()) if k.startswith("W")]
bs = [data[k] for k in sorted(data.keys()) if k.startswith("b") and not k.endswith("x_mean") and not k.endswith("y_mean")]
n_cp = 6

def ann_numpy_forward(X_raw):
  H = (X_raw - x_mean) / x_scale
  n_layers = len(Ws)
  for layer in range(n_layers):
    W = Ws[layer]
    b = bs[layer]
    H = H @ W.T + b
    if layer < n_layers - 1:
      H = np.tanh(H)
  Y_raw = H * y_scale + y_mean
  Y_raw[:, :n_cp] = np.exp(Y_raw[:, :n_cp])
  return Y_raw

def run_ann(args):
  T, F_feed, p0, pp, c_feed = args
  c_feed = np.array(c_feed)

  # input_names = ["T","F_f","P_f","P_p","C0_Ca","C0_Mg","C0_Na","C0_K","C0_SO4","C0_Cl"]
  X_raw = np.array([T, F_feed, p0, pp, *c_feed]).reshape(1, -1)

  # ANN prediction
  # output_names = ["Cp_Ca","Cp_Mg","Cp_Na","Cp_K","Cp_SO4","Cp_Cl","F_p","P_r"]
  Y_ann = ann_numpy_forward(X_raw).flatten()

  ns = len(c_feed)
  # Extract outputs
  Cp = Y_ann[:ns]
  Fp = Y_ann[-2]
  Pr = Y_ann[-1]

  Fr = max(F_feed - Fp, 1e-12)
  Cr = (F_feed * c_feed - Fp * Cp) / Fr

  els = {}
  els['Fr'] = Fr
  els['cr'] = Cr
  els['pr'] = Pr
  els['F0'] = F_feed
  els['c0'] = c_feed
  els['p0'] = p0
  els['Fp'] = Fp
  els['cp'] = Cp
  els['pp'] = pp
  return els
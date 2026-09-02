# OptMemPy
Optimization framework for nanofiltration cascade systems using hybrid mechanistic-surrogate simulations.

Created by: Kenia Moreno Martinez

---

## Overview

OptMemPy is a Python-based framework for:
- hybrid model simulation of nanofiltration (NF) cascade systems
- multi-objective local optimization using IPOPT
- multi-objective global optimization using MAiNGO
- Pareto front generation and simulation validation

The framework was developed for the study of multi-stage NF cascade systems for selective ion recovery and desalination applications under different operating temperatures.

This repository contains code supporting the manuscript:

"Optimization of nanofiltration cascades for brine separation using temperature-dependent mechanistic and data-driven surrogate models"  
K. Moreno Martinez et al. (DOI:XX/XXX)

The multi-start multi-objective local optimization framework based on Pyomo/IPOPT builds upon the work developed by A. K. Beke et al. in:

"Multiobjective optimization and process intensification of multistage nanofiltration with structural synergy for brine separation"  
DOI:10.1016/j.cej.2024.158994

and the associated GitHub repository:
https://github.com/aronbeke/meprom-opti

---

## Repository Structure

```text
optmempy/
│
├── environment.yml
├── requirements.txt
├── README.md
│
├── optmempy/
│   │
│   ├── nf_cascade_simulation/
│   │   ├── ANN membrane-element simulation
│   │   ├── Hybrid NF cascade process model simulation
│   │
│   ├── nf_cascade_ann/
│   │   ├── NF cascade process ANN surrogate model for MAiNGO's framework
│   │
│   ├── ipopt_framework/
│   │   ├── local multi-objective optimization
│   │
│   ├── maingo_framework/
│   │   ├── global multi-objective ANN-based optimization
```

---

## Installation

### 1. Clone repository

```bash
git clone <repository-url>
cd optmempy
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate environment

```bash
conda activate optmempy
```

---

## IPOPT Optimization Framework

The IPOPT framework performs multi-objective optimization of NF cascade systems using hybrid mechanistic-surrogate simulation models.

### Run IPOPT optimization

From the repository root:

```bash
python -m optmempy.ipopt_framework.run_optimization
```

The script will prompt the user for an input `.txt` file.

---

## IPOPT Input File

Example:

```text
objective=molar_power
max_pressure=22.4
no_models=5
constraints=1.2,1.5
pressure_exchange=1
n_stages=6
relax=1
temperature=303
transport_model=sdec
```

### Parameters

| Parameter | Description |
|---|---|
| objective | `separation_factor` or `molar_power` |
| max_pressure | Maximum feed pressure [bar] |
| no_models | Number of multistart initializations |
| constraints | Pareto constraint levels |
| pressure_exchange | Enable pressure exchange (0/1) |
| n_stages | Number of NF stages |
| relax | Enable relaxation variables (0/1) |
| temperature | Feed temperature [K] |
| transport_model | Transport model identifier |

- If the objective is `separation_factor`, pareto constraint is the minimal magnesium ion recovery.
- If the objective is `molar_power`, pareto constraint is the minimal separataion factor.
- `constraints`. Recovery or separation factor values separated by commas.
- `pressure_exchange`. 0: Does not include pressure exchange. 1: Includes pressure exchange.
- `relax`. 0: MINLP. 1: NLP.
- If not specified, the optimization framework will randomly initialize.

---

## User-Defined Initialization

The IPOPT framework also supports user-defined initialization through CSV files.

Example input file:

```text
objective=molar_power
max_pressure=40
no_models=3
constraints=1.2,1.5
pressure_exchange=1
n_stages=3
relax=1
temperature=303
transport_model=sdec
initialization_file=ipopt_initialization_example.csv
```

- The number of rows is the number of initializations given by the user.
- The CSV file must contain one initialization per row.
- The number of rows in the CSV file must match `no_models`.
- Each row must contain all variables required to reconstruct a complete `process_parameters` dictionary:
```text
process_parameters = {
  'p_feed',
  'pp_list',
  'LambdaNorm_mx',
  'PiNorm_mx',
  'dNorm_vector',
  'fNorm_vector',
}
```
- The variables are flattened into CSV columns as:

| Variable        | Number of columns |
| --------------- | ----------------- |
| `p_feed`        | 1                 |
| `pp_list`       | `n_stages`        |
| `LambdaNorm_mx` | `n_stages²`       |
| `PiNorm_mx`     | `n_stages²`       |
| `dNorm_vector`  | `n_stages`        |
| `fNorm_vector`  | `n_stages`        |

- The total number of required CSV columns is: `n_columns = 1 + 3·(n_stages) + 2·(n_stages²)`. For example, for `n_stages=3`, the CSV file must contain 28 columns.
- Example column ordering for `n_stages=3`:
```text
p_feed,
pp_0, pp_1, pp_2,
LambdaNorm_0_0, ..., LambdaNorm_2_2,
PiNorm_0_0, ..., PiNorm_2_2,
dNorm_0, dNorm_1, dNorm_2,
fNorm_0, fNorm_1, fNorm_2
```
- The framework will validate pressure bounds, non-negative values, normalization constraints, matrix dimensions, and no self-loop recycle (`LambdaNorm_i_i = 0` and `PiNorm_i_i = 0`).

- IPOPT solver verbosity and settings can be modified from `ipopt_framework/opti_model.py` in the `optimize` module.

---

## MAiNGO Optimization Framework

The MAiNGO framework performs global optimization using ANN surrogate models of the NF cascade system.

### Run MAiNGO optimization

```bash
python -m optmempy.maingo_framework.run_optimization
```

The script will prompt the user for an input `.txt` file.

---

## MAiNGO Input File

Example:

```text
objective=separation_factor
max_pressure=40
constraints=0.93,0.94
n_stages=6
temperature=303
plot_pareto=1
```

- If the objective is `separation_factor`, pareto constraint is the minimal magnesium ion recovery.
- If the objective is `molar_power`, pareto constraint is the minimal separataion factor.
- `constraints`. Recovery or separation factor values separated by commas.
- `plot_pareto`. 0: No plotting. 1: Plots objective optima values against constraint values.
- MAiNGO solver verbosity can be modified from `maingo_framework/run_optimization.py`. Default = `False`.
- MAiNGO solver settings can be modified from `maingo_framework/run_maingo.py`

### ANN surrogate model
The ANN embedded into this framework is found in `nf_cascade_ann` folder.
The ANN model describes a specific NF cascade system superstructure. (See DOI:XX/XXX Section X.X)
All bounds, constraints and objectives are defined in accordance to this ANN's structure.
The user can use their own ANN model in a `.xml` format, with their corresponding bounds, constraints and objectives.

---

## Output Files

Optimization results are automatically saved in the corresponding `results/` folders in CSV format.

Typical outputs include:
- all optimization runs
- validated solutions
- Pareto-optimal solutions
- performance metrics
- simulation warnings

---

## Notes

- ANN surrogate validity is limited to the training domain.
- MAiNGO's optimal solutions are dependent on the physical validity of the specified surrogate model.
- Users are encouraged to validate ANN-based optimal solutions using mechanistic or hybrid simulations.

---

## Citation

If you use this repository in academic work, please cite:

```text
[CITATION]
```

---

## References

1. [This paper]

2. A.K. Beke, S. Ihm, F.A. Alharthi, C.M. Fellows, G. Szekely, Multi-objective optimization and process intensification of multistage nanofiltration with structural synergy for brine separation, Chemical Engineering Journal 505 (2025) 158994. https://doi.org/10.1016/j.cej.2024.158994.

3. memprom-opti GitHub repository: https://github.com/aronbeke/meprom-opti

4. M.L. Bynum, G.A. Hackebeil, W.E. Hart, C.D. Laird, B.L. Nicholson, J.D. Siirola, J.-P. Watson, D.L. Woodruff, Pyomo — Optimization Modeling in Python, Springer International Publishing, Cham, 2021. https://doi.org/10.1007/978-3-030-68928-5.

5. A. Wächter, L.T. Biegler, On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming, Math. Program. 106 (2006) 25–57. https://doi.org/10.1007/s10107-004-0559-y.

6. D. Bongartz, J. Najman, S. Sass, A. Mitsos, McCormick-based Algorithm for mixed-integer Nonlinear Global Optimization, RWTH Aachen University, 2018. http://permalink.avt.rwth-aachen.de/?id=729717.

---

## Contact

Kenia Moreno Martinez. 
King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia. 
kenia.morenomartinez@kaust.edu.sa

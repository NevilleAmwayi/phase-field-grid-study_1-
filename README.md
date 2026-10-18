# Grid-size study for a JAX phase-field code

How grid resolution affects runtime, memory and simulated patterns for a 2-D Cahn-Hilliard model
(Landau free energy, explicit Euler, float32, CPU).

This repository contains **only my benchmark scripts, results and report**. The simulation code is
the third-party package [`ajvetturini/phase-field-models`](https://github.com/ajvetturini/phase-field-models)

## Contents
| File | Purpose |
|---|---|
| `bench.py` | Runs one simulation or timing at a single grid size (fresh process, clean memory numbers) |
| `run_all.py` | Driver: `timing` (per-step cost, N = 16 to 1024) and `sim` (identical physical run, N = 32 to 256) |
| `make_report.py` | Builds the tables (CSV / markdown) and plots from `results/` |
| `report.md`, `report.pdf` | Short report on the cost vs. resolution trade-off |
| `results/` | Tables, plots and raw outputs from my run (2 vCPU CPU, JAX 0.11.2, Python 3.13) |

## Reproduce

**Windows (PowerShell)**
```powershell
git clone https://github.com/ajvetturini/phase-field-models ..\phase-field-models
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PFM_REPO = "..\phase-field-models"
python run_all.py timing
python run_all.py sim
python make_report.py
```

**Mac / Linux**
```bash
git clone https://github.com/ajvetturini/phase-field-models ../phase-field-models
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PFM_REPO=../phase-field-models
python run_all.py timing
python run_all.py sim
python make_report.py
```

## Headline results (CPU only)
- Time per step scales roughly as N^1.9. With explicit Euler, `dt` must scale as `dx^4`, so total cost grows about N^6 once the stability limit binds (N >= 256).
- N = 128 gives about 2% energy error versus N = 256 at roughly 3.6x lower cost.
- N = 32 is under-resolved (grid-scale noise); N = 64 has about 10% energy error.

Limitations: one initial condition, one model/integrator, float32, CPU timings only; N = 256 is the
reference, not a converged truth. See `report.pdf`.

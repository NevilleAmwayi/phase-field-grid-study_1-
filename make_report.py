"""Builds the runtime table (CSV + markdown) and all plots from results/*.json|npz."""
import json, numpy as np, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = 'results/'
tim = json.load(open(R + 'timing.json')); sim = json.load(open(R + 'sim.json'))
L, DT, T_END = 128.0, 2.5e-4, 200.0
base_rss = min(r['peak_rss_mb'] for r in tim)

# ---- Table 1: per-step cost vs N ----
tab1 = []
for r in tim:
    N = r['N']; dx = L / N
    dt_max = dx ** 4 / (32 * 2.0 * 1.0 * 1.0 / 2.0)          # explicit-Euler limit dx^4/(32*M*kappa), kappa=interface_scalar*k=2
    dt_max = dx ** 4 / (32 * 2.0)
    dt_used = min(DT, 0.25 * dt_max)                          # stability-limited (25% of limit) but never above the dt used in Exp 2
    steps_T = T_END / dt_used
    tab1.append(dict(N=N, cells=N * N, dx=dx, us_step=r['us_per_step'], ns_cell=r['ns_per_cell_step'],
                     compile_s=r['compile_s'], field_kb=r['field_bytes'] / 1024,
                     xla_temp_kb=r.get('xla_temp_bytes', 0) / 1024,
                     rss_mb=r['peak_rss_mb'], rss_extra_mb=r['peak_rss_mb'] - base_rss,
                     dt_stable=dt_used, steps_to_T=steps_T, est_hours=steps_T * r['us_per_step'] * 1e-6 / 3600))
with open(R + 'table_cost_vs_N.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(tab1[0].keys())); w.writeheader(); w.writerows(tab1)

# ---- Table 2: identical simulation (same dt, same T) ----
ref = np.load(R + 'sim_N256.npz'); refs = ref['snaps'][-1].astype(np.float64)
tab2 = []
for r in sim:
    N = r['N']; d = np.load(R + f'sim_N{N}.npz'); s = d['snaps'][-1].astype(np.float64)
    k = 256 // N
    refc = refs.reshape(N, k, N, k).mean(axis=(1, 3))        # block-average reference down to N x N
    err = np.linalg.norm(s - refc) / np.linalg.norm(refc)
    tab2.append(dict(N=N, dx=L / N, steps=r['steps'], wall_s=r['wall_s'], rss_mb=r['peak_rss_mb'],
                     energy=d['energy'][-1], energy_err_pct=100 * (d['energy'][-1] - ref['energy'][-1]) / abs(ref['energy'][-1]),
                     length=d['length'][-1], mean=d['mean'][-1], rel_L2_err_vs_256=err,
                     pmin=r['final']['pmin'], pmax=r['final']['pmax']))
with open(R + 'table_simulation.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(tab2[0].keys())); w.writeheader(); w.writerows(tab2)

def md(rows, cols):
    h = '| ' + ' | '.join(c[0] for c in cols) + ' |\n|' + '---|' * len(cols) + '\n'
    return h + '\n'.join('| ' + ' | '.join(c[1](r) for c in cols) + ' |' for r in rows)
open(R + 'tables.md', 'w').write(
    '### Table 1: cost per step\n' + md(tab1, [('N', lambda r: str(r['N'])), ('dx (nm)', lambda r: f"{r['dx']:.3g}"),
    ('µs/step', lambda r: f"{r['us_step']:.1f}"), ('ns/cell/step', lambda r: f"{r['ns_cell']:.1f}"),
    ('JIT compile (s)', lambda r: f"{r['compile_s']:.2f}"), ('field (KB)', lambda r: f"{r['field_kb']:.0f}"),
    ('XLA temp (KB)', lambda r: f"{r['xla_temp_kb']:.0f}"), ('peak RSS (MB)', lambda r: f"{r['rss_mb']:.0f}"),
    ('stable dt', lambda r: f"{r['dt_stable']:.2e}"), ('est. hours to T=200', lambda r: f"{r['est_hours']:.3g}")]) +
    '\n\n### Table 2: identical simulation (dt=2.5e-4, T=200)\n' + md(tab2, [('N', lambda r: str(r['N'])),
    ('dx (nm)', lambda r: f"{r['dx']:.3g}"), ('steps', lambda r: f"{r['steps']:,}"), ('runtime (s)', lambda r: f"{r['wall_s']:.1f}"),
    ('peak RSS (MB)', lambda r: f"{r['rss_mb']:.0f}"), ('energy/area', lambda r: f"{r['energy']:.4f}"),
    ('energy err vs 256 (%)', lambda r: f"{r['energy_err_pct']:+.1f}"), ('domain length (nm)', lambda r: f"{r['length']:.2f}"),
    ('rel. L2 error vs 256', lambda r: f"{r['rel_L2_err_vs_256']:.3f}"), ('mean φ', lambda r: f"{r['mean']:.4f}")]))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
Ns = np.array([r['N'] for r in tab1]); us = np.array([r['us_step'] for r in tab1])
# Plot 1: per-step cost + scaling exponent
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(Ns, us, 'o-', color='#1f5fa8', label='measured')
big = Ns >= 64; p = np.polyfit(np.log(Ns[big]), np.log(us[big]), 1)
ax[0].loglog(Ns, np.exp(p[1]) * Ns ** 2, '--', color='gray', label='ideal $N^2$')
ax[0].set(xlabel='grid size N (N×N cells)', ylabel='time per step (µs)', title=f'Cost per step (fit slope N≥64: {p[0]:.2f})'); ax[0].legend()
ax[1].plot(Ns, [r['ns_cell'] for r in tab1], 'o-', color='#c8501e')
ax[1].set_xscale('log'); ax[1].set(xlabel='grid size N', ylabel='ns per cell per step', title='Efficiency per cell (CPU cache effects)')
fig.tight_layout(); fig.savefig(R + 'plot_runtime_per_step.png', dpi=150); plt.close(fig)
# Plot 2: memory
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(Ns, [r['field_kb'] for r in tab1], 'o-', label='one field (float32)')
ax.loglog(Ns, [max(r['xla_temp_kb'], 1) for r in tab1], 's-', label='XLA temporaries / scan')
ax.loglog(Ns, [max(r['rss_extra_mb'], 0.01) * 1024 for r in tab1], '^--', label='peak RSS above N=16 baseline')
ax.set(xlabel='grid size N', ylabel='KB', title='Memory vs grid size'); ax.legend(fontsize=8)
fig.text(0.01, 0.01, f'baseline process (JAX runtime) ≈ {base_rss:.0f} MB', fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(R + 'plot_memory.png', dpi=150); plt.close(fig)
# Plot 3: cost to reach T=200 with the stability-limited dt
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(Ns, [r['est_hours'] * 3600 for r in tab1], 'o-', color='#7a3e9d', label='estimated (dt = min(2.5e-4, 25% stability limit))')
ax.loglog([r['N'] for r in tab2], [r['wall_s'] for r in tab2], 's', color='k', label='measured (fixed dt=2.5e-4)')
sel = Ns >= 256
p6 = np.polyfit(np.log(Ns[sel]), np.log([r['est_hours'] * 3600 for r in tab1])[sel], 1)
ax.set(xlabel='grid size N', ylabel='seconds to simulate T=200', title=f'Total cost (slope for N≥256 ≈ {p6[0]:.1f})'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(R + 'plot_total_cost.png', dpi=150); plt.close(fig)
# Plot 4: patterns at T=200 and at an early time
fig, axes = plt.subplots(2, len(sim), figsize=(3.1 * len(sim), 6.4))
for j, r in enumerate(sim):
    d = np.load(R + f"sim_N{r['N']}.npz")
    for i, idx in enumerate([5, -1]):
        axx = axes[i, j]; axx.imshow(d['snaps'][idx], cmap='RdBu_r', vmin=-1, vmax=1, extent=[0, L, 0, L], interpolation='nearest')
        axx.set_title(f"N={r['N']} (dx={L/r['N']:.2g}), t={d['times'][idx]:.0f}", fontsize=9); axx.set_xticks([]); axx.set_yticks([])
fig.tight_layout(); fig.savefig(R + 'plot_patterns.png', dpi=150); plt.close(fig)
# Plot 5: energy/length-scale histories + final error
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for r in sim:
    d = np.load(R + f"sim_N{r['N']}.npz")
    ax[0].plot(d['times'], d['energy'], label=f"N={r['N']}"); ax[1].plot(d['times'], d['length'], label=f"N={r['N']}")
ax[0].set(xlabel='time', ylabel='free energy per area', title='Energy decay'); ax[1].set(xlabel='time', ylabel='characteristic length (nm)', title='Coarsening length'); ax[0].legend(fontsize=8)
ax[2].loglog([r['N'] for r in tab2[:-1]], [r['rel_L2_err_vs_256'] for r in tab2[:-1]], 'o-'); ax[2].set(xlabel='N', ylabel='rel. L2 error vs N=256', title='Pattern difference at final time')
for a_ in ax: a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(R + 'plot_energy_length_error.png', dpi=150); plt.close(fig)
print(open(R + 'tables.md').read()); print('fit per-step slope', p[0], 'total-cost slope', p6[0])

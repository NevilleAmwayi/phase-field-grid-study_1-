"""Driver: Exp 1 = per-step cost vs N (timing); Exp 2 = identical physical simulation at several N."""
import json, subprocess, sys, os
REPO = os.environ.get('PFM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase-field-models'))
env = dict(os.environ, PYTHONPATH=REPO)
def run(args):
    out = subprocess.run([sys.executable, 'bench.py'] + args, env=env, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith('{')]
    if not line:
        print('FAILED', args, out.stderr[-400:]); return None
    return json.loads(line[-1])

which = sys.argv[1]
if which == 'timing':
    rows = []
    for N in [16, 32, 64, 128, 256, 512, 1024]:
        steps = int(min(20000, max(100, 3e7 / N ** 2)))
        r = run(['timing', '--N', str(N), '--dt', '1e-5', '--steps', str(steps)])
        if r: rows.append(r); print(N, round(r['us_per_step'], 1), 'us/step', round(r['peak_rss_mb']), 'MB', flush=True)
    json.dump(rows, open('results/timing.json', 'w'), indent=1)
else:
    rows = []
    for N in [32, 64, 128, 256]:
        r = run(['sim', '--N', str(N), '--dt', '2.5e-4', '--T', '200', '--out', f'results/sim_N{N}.npz'])
        if r: rows.append(r); print(N, round(r['wall_s'], 1), 's', r['final'], flush=True)
        json.dump(rows, open('results/sim.json', 'w'), indent=1)

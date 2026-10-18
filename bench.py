"""Worker: one Cahn-Hilliard (Landau, explicit Euler) run at a single grid size.
Run in its own process so peak memory (RSS) is clean.  Usage:
  python bench.py timing --N 128 --dt 1e-5 --steps 2000
  python bench.py sim    --N 128 --dt 2.5e-4 --T 150 --out results/sim_N128.npz
"""
import argparse, json, os, sys, time
import numpy as np

L = 128.0            # physical domain length (same for every N -> dx = L/N)
EPS, KAPPA_K, IFACE, MOB = 0.9, 1.0, 2.0, 1.0   # Landau eps, k, interface_scalar, M
MEAN, AMP, KMAX = 0.01, 0.02, 8                  # initial condition

def peak_rss_mb():
    """Peak memory of this process in MB (Linux/Mac via resource, Windows via psutil)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1048576

def make_cfg(N, dt):
    return dict(N=N, dt=dt, dx=L / N, free_energy='landau', integrator='euler', model='ch',
                use_autodiff=False, interface_scalar=IFACE, k=KAPPA_K, M=MOB,
                float_type='float32', dim=2, landau={'epsilon': EPS},
                custom_initial_condition=True, steps=1, verbose=False, seed=0)

def init_fn(field):
    """Same smooth random field (modes |m|<=8 on a fixed physical domain) at every N."""
    N = field.shape[-1]
    rng = np.random.default_rng(1234)
    m = np.arange(-KMAX, KMAX + 1)
    a = rng.normal(size=(len(m), len(m))); b = rng.normal(size=(len(m), len(m)))
    a[KMAX, KMAX] = b[KMAX, KMAX] = 0.0
    x = np.arange(N) * L / N
    ph = 2 * np.pi * np.outer(m, x) / L
    c, s = np.cos(ph), np.sin(ph)
    phi = (np.einsum('ab,ai,bj->ij', a, c, c) - np.einsum('ab,ai,bj->ij', a, s, s)
           + np.einsum('ab,ai,bj->ij', b, s, c) + np.einsum('ab,ai,bj->ij', b, c, s))
    std = np.sqrt(((a ** 2 + b ** 2) / 2).sum())    # analytic std -> independent of N
    field[0] = MEAN + AMP * phi / std
    return field

def metrics(rho, dx):
    p = np.asarray(rho[0], dtype=np.float64)
    gx = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / (2 * dx)
    gy = (np.roll(p, -1, 1) - np.roll(p, 1, 1)) / (2 * dx)
    f = -0.5 * EPS * p ** 2 + 0.25 * p ** 4 + KAPPA_K * (gx ** 2 + gy ** 2)
    N = p.shape[0]
    S = np.abs(np.fft.fft2(p - p.mean())) ** 2
    q = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    qm = np.sqrt(q[:, None] ** 2 + q[None, :] ** 2)
    qbar = (qm * S).sum() / max(S.sum(), 1e-300)
    return dict(energy=float(f.mean()), mean=float(p.mean()), std=float(p.std()),
                length=float(2 * np.pi / qbar) if qbar > 0 else float('nan'),
                pmin=float(p.min()), pmax=float(p.max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['timing', 'sim'])
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--dt', type=float, required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--T', type=float, default=100.0)
    ap.add_argument('--nint', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import jax, jax.numpy as jnp
    from pfm.manager import SimulationManagerNoWrite
    mgr = SimulationManagerNoWrite(make_cfg(a.N, a.dt), custom_initial_condition=init_fn)
    system = mgr._system
    rho0 = jnp.asarray(mgr.init_field, dtype=jnp.float32)
    dx = L / a.N

    def make_fn(n):
        def run(r):
            r, _ = jax.lax.scan(lambda c, _: (system.evolve(c), None), r, None, length=n)
            return r
        return jax.jit(run)

    if a.mode == 'timing':
        fn = make_fn(a.steps)
        t0 = time.perf_counter(); fn(rho0).block_until_ready(); first = time.perf_counter() - t0
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); out = fn(rho0).block_until_ready(); ts.append(time.perf_counter() - t0)
        run_t = min(ts)
        mem = {}
        try:
            ma = fn.lower(rho0).compile().memory_analysis()
            mem = dict(xla_temp_bytes=int(ma.temp_size_in_bytes), xla_arg_bytes=int(ma.argument_size_in_bytes),
                       xla_out_bytes=int(ma.output_size_in_bytes))
        except Exception as e:
            mem = dict(xla_error=str(e)[:80])
        res = dict(N=a.N, dx=dx, dt=a.dt, steps=a.steps, first_call_s=first, run_s=run_t,
                   compile_s=max(first - run_t, 0.0), us_per_step=1e6 * run_t / a.steps,
                   ns_per_cell_step=1e9 * run_t / a.steps / a.N ** 2,
                   field_bytes=a.N ** 2 * 4, peak_rss_mb=peak_rss_mb(),
                   finite=bool(np.isfinite(np.asarray(out)).all()), **mem)
        print(json.dumps(res)); return

    # ---- sim mode ----
    total = int(round(a.T / a.dt)); chunk = total // a.nint; total = chunk * a.nint
    fn = make_fn(chunk)
    fn(rho0).block_until_ready()                      # warm-up / compile (excluded from runtime)
    t_start = time.perf_counter()
    r = rho0
    times, series, snaps = [0.0], [metrics(rho0, dx)], [np.asarray(rho0[0], np.float32)]
    for i in range(a.nint):
        r = fn(r); r.block_until_ready()
        times.append((i + 1) * chunk * a.dt)
        series.append(metrics(r, dx)); snaps.append(np.asarray(r[0], np.float32))
    wall = time.perf_counter() - t_start
    res = dict(N=a.N, dx=dx, dt=a.dt, steps=total, T=total * a.dt, wall_s=wall,
               peak_rss_mb=peak_rss_mb(),
               finite=bool(np.isfinite(snaps[-1]).all()), final=series[-1])
    if a.out:
        np.savez_compressed(a.out, snaps=np.array(snaps), times=np.array(times),
                            energy=[s['energy'] for s in series], mean=[s['mean'] for s in series],
                            std=[s['std'] for s in series], length=[s['length'] for s in series])
    print(json.dumps(res))

if __name__ == '__main__':
    main()

"""Monotone finite-grid Markov-chain HJB approximation, not an exact solution.

Run spatial, temporal, and action refinement before treating outputs as a
numerical reference. All spatial coordinates below are normalized to [0,1].
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from model import coefficients, terminal, load_config


def solve(c, nodes=13, context_nodes=9, controls=21, dt_max=0.05):
    assert nodes >= 3 and context_nodes >= 3 and controls >= 2 and dt_max > 0
    axes = [np.linspace(0,1,nodes), np.linspace(0,1,nodes),
            np.linspace(0,1,context_nodes)]
    mesh = np.stack(np.meshgrid(*axes,indexing='ij'), axis=-1)
    physical = mesh*np.array([c.K1,c.K2,1.0])
    hs = [a[1]-a[0] for a in axes]
    scales = [c.K1,c.K2,1.0]
    steps = int(np.ceil(c.T/dt_max))
    dt = c.T/steps
    times = np.linspace(0,c.T,steps+1)
    actions = np.linspace(0,1,controls)
    shape = mesh.shape[:-1]
    values = np.empty((steps+1,)+shape)
    policies = np.empty((steps,)+shape)
    values[-1] = terminal(physical,c)
    max_cfl = 0.0
    for k in range(steps-1,-1,-1):
        v = values[k+1]
        best = np.full(shape,np.inf)
        policy = np.zeros(shape)
        for u in actions:
            fs, ss, cost, _ = coefficients(physical,times[k],u,c)
            qsum = np.zeros(shape)
            generator = np.zeros(shape)
            for axis in range(3):
                b, s = fs[axis]/scales[axis], ss[axis]/scales[axis]
                qp = np.maximum(b,0)/hs[axis]+0.5*s*s/hs[axis]**2
                qm = np.maximum(-b,0)/hs[axis]+0.5*s*s/hs[axis]**2
                hi, lo = [slice(None)]*3, [slice(None)]*3
                hi[axis], lo[axis] = -1, 0
                if np.max(qp[tuple(hi)]) > 1e-10 or np.max(qm[tuple(lo)]) > 1e-10:
                    raise ValueError('Outward boundary transition: inspect model invariance')
                qp[tuple(hi)], qm[tuple(lo)] = 0, 0
                generator += qp*(np.roll(v,-1,axis)-v)+qm*(np.roll(v,1,axis)-v)
                qsum += qp+qm
            cfl = float(dt*qsum.max())
            max_cfl = max(max_cfl,cfl)
            if cfl > 1+1e-12:
                raise ValueError(f'CFL={cfl:.3f} exceeds 1; decrease --dt')
            candidate = v+dt*(cost+generator)
            take = candidate < best
            best[take], policy[take] = candidate[take], u
        values[k], policies[k] = best, policy
    return dict(values=values, policies=policies, times=times,
                y1=axes[0], y2=axes[1], x=axes[2], max_cfl=max_cfl,
                config_json=json.dumps(asdict(c)))


def reference_policy(path, c):
    data = np.load(path)
    if json.loads(str(data['config_json'])) != asdict(c):
        raise ValueError('Reference configuration differs from simulation configuration')
    # Piecewise constant in time, trilinear in the state. This interpolated
    # policy is approximate and is evaluated independently in the simulator.
    times = data['times']
    policies = data['policies']
    axes = (data['y1'], data['y2'], data['x'])
    cache = {}
    def policy(z,t):
        k = int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2))
        if k not in cache:
            cache.clear()
            cache[k] = RegularGridInterpolator(axes,policies[k], bounds_error=True)
        return np.clip(cache[k](z/np.array([c.K1,c.K2,1.0])),0,1)
    return policy


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--nodes',type=int,default=13)
    p.add_argument('--context-nodes',type=int,default=9)
    p.add_argument('--controls',type=int,default=21)
    p.add_argument('--dt',type=float,default=0.05)
    p.add_argument('--output',default='results/reference.npz')
    a = p.parse_args(); c = load_config(a.config)
    result = solve(c,a.nodes,a.context_nodes,a.controls,a.dt)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**result)
    print(json.dumps(dict(output=a.output,max_cfl=result['max_cfl'],
                         shape=list(result['values'].shape)),indent=2))

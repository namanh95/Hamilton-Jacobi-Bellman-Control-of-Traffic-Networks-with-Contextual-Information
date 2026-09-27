"""Deterministic mathematical checks; these do not establish PINN performance."""
import ast
import json
from pathlib import Path
import numpy as np
from model import Config, coefficients, feedback
from reference import solve
from evaluate import simulate


def main():
    c=Config();rng=np.random.default_rng(942)
    upper=np.array([c.K1,c.K2,1.]);z=rng.uniform(size=(200,3))*upper
    t=rng.uniform(0,c.T,200);u=rng.uniform(size=200)
    f,s,L,flow=coefficients(z,t,u,c)
    balance=float(np.max(np.abs(f[0]+f[1]-(flow['offered']-flow['rejected']-flow['completed']))))
    assert balance<1e-10
    face_worst=0.
    for axis in range(3):
        for side in [0,1]:
            zz=z.copy();zz[:,axis]=side*upper[axis]
            ff,ss,_,_=coefficients(zz,t,u,c)
            assert np.max(np.abs(ss[axis]))<1e-12
            outward= -ff[axis] if side==0 else ff[axis]
            face_worst=max(face_worst,float(outward.max()))
            assert outward.max()<1e-10
    gradients=rng.normal(0,.02,(200,3))
    opt=feedback(gradients,z,c)
    A=flow['transfer'];b=A*(gradients[:,0]-gradients[:,1])
    h=c.ru*opt**2+b*opt
    grid=np.linspace(0,1,10001)
    discrete=(c.ru*grid[None,:]**2+b[:,None]*grid).min(1)
    assert np.max(h-discrete)<1e-10
    near=gradients+rng.normal(0,.005,(200,3));u2=feedback(near,z,c)
    err=(near[:,1]-near[:,0])-(gradients[:,1]-gradients[:,0])
    regret=c.ru*u2**2+b*u2-h
    assert np.max(regret-A*A*err*err/(4*c.ru))<1e-10
    ref=solve(Config(T=.2),nodes=5,context_nodes=5,controls=5,dt_max=.02)
    assert np.isfinite(ref['values']).all() and ref['values'].min()>=0
    metrics,_=simulate(Config(T=1),lambda z,t:np.ones(len(z)),[1700,1500,.05],8,.05,123)
    assert metrics['balance_error_max'].max()<1e-8
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
    report=dict(flow_balance_max_abs=balance,maximum_outward_face_drift=face_worst,
                analytic_control_check='PASS',hamiltonian_regret_check='PASS',
                reference_smoke_max_cfl=ref['max_cfl'],
                simulation_balance_max_abs=float(metrics['balance_error_max'].max()),
                python_syntax='PASS',pinn_training='NOT RUN',publication_experiments='NOT RUN')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

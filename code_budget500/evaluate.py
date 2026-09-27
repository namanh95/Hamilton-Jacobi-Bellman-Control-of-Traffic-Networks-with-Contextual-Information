"""Paired Monte Carlo evaluation with explicit demand and projection accounting."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from model import coefficients, terminal, feedback, load_config


def simulate(c,policy,initial,paths=200,dt=0.1,seed=1234):
    if paths<2 or dt<=0:raise ValueError('At least two paths and a positive time step required')
    steps=int(np.ceil(c.T/dt));dt=c.T/steps
    rng=np.random.default_rng(seed)
    z=np.tile(np.asarray(initial,float),(paths,1));upper=np.array([c.K1,c.K2,1.0])
    if np.any(z<0) or np.any(z>upper):raise ValueError('Initial state outside model domain')
    names=['cost','vehicle_hours','completed','rejected','offered','max_core','max_total',
           'saturation','projection_vehicle_abs','projection_context_abs','projection_steps',
           'balance_error_max']
    metrics={k:np.zeros(paths) for k in names}
    metrics['max_core']=z[:,0].copy();metrics['max_total']=z[:,:2].sum(1)
    trajectory=[]
    for k in range(steps):
        t=k*dt;u=np.broadcast_to(np.asarray(policy(z,t)),(paths,))
        if np.any(u<0) or np.any(u>1):raise ValueError('Inadmissible control')
        f,s,cost,flows=coefficients(z,t,u,c)
        metrics['cost']+=cost*dt
        metrics['vehicle_hours']+=z[:,:2].sum(1)*dt/60
        for name in ['completed','rejected','offered']:metrics[name]+=flows[name]*dt
        q=np.quantile(z,[.025,.5,.975],axis=0)
        trajectory.append([t,*q[0],*q[1],*q[2],float(np.mean(u))])
        noise=np.stack(s,axis=1)*np.sqrt(dt)*rng.standard_normal((paths,3))
        raw=z+dt*np.stack(f,axis=1)+noise
        next_z=np.clip(raw,0,upper); correction=next_z-raw
        metrics['projection_vehicle_abs']+=np.abs(correction[:,:2]).sum(1)
        metrics['projection_context_abs']+=np.abs(correction[:,2])
        metrics['projection_steps']+=np.any(correction!=0,axis=1)
        balance=(next_z[:,:2]-z[:,:2]).sum(1)-(flows['offered']-flows['rejected']-flows['completed'])*dt-noise[:,:2].sum(1)-correction[:,:2].sum(1)
        metrics['balance_error_max']=np.maximum(metrics['balance_error_max'],np.abs(balance))
        z=next_z
        metrics['max_core']=np.maximum(metrics['max_core'],z[:,0])
        metrics['max_total']=np.maximum(metrics['max_total'],z[:,:2].sum(1))
        metrics['saturation']=np.maximum(metrics['saturation'],np.any(z[:,:2]>=.95*upper[:2],axis=1))
    metrics['cost']+=terminal(z,c)
    q=np.quantile(z,[.025,.5,.975],axis=0)
    trajectory.append([c.T,*q[0],*q[1],*q[2],float('nan')])
    return metrics,np.asarray(trajectory)


def neural_policy(checkpoint,c):
    import torch
    from train_pinn import ValueNet,DTYPE
    # Only load checkpoints you created or trust; torch files may contain pickle.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['config']!=asdict(c):raise ValueError('Checkpoint model mismatch')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    net=ValueNet(c,saved.get('ansatz','soft'));net.load_state_dict(saved['state']);net.eval()
    def policy(z,t):
        p=torch.tensor(np.column_stack([z,np.full(len(z),t)]),dtype=DTYPE,requires_grad=True)
        g=torch.autograd.grad(net(p).sum(),p)[0]
        return feedback(g,p[:,:3],c,torch).detach().numpy()
    return policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--reference')
    p.add_argument('--paths',type=int,default=200);p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--seed',type=int,default=1234);p.add_argument('--gain',type=float,default=2)
    p.add_argument('--initial',type=float,nargs=3,default=[1700,1500,.05])
    p.add_argument('--output',default='results/evaluation');a=p.parse_args()
    c=load_config(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    policies=dict(open=lambda z,t:np.ones(len(z)),fixed=lambda z,t:np.full(len(z),.5),
                  reactive=lambda z,t:np.clip(1-a.gain*np.maximum(z[:,0]/c.K1-.5,0),0,1))
    if a.checkpoint:policies['pinn']=neural_policy(a.checkpoint,c)
    if a.reference:
        from reference import reference_policy
        policies['grid']=reference_policy(a.reference,c)
    records=[];summaries={};allmetrics={}
    for name,policy in policies.items():
        m,tr=simulate(c,policy,a.initial,a.paths,a.dt,a.seed);allmetrics[name]=m
        summaries[name]={k:dict(mean=float(v.mean()),sd=float(v.std(ddof=1))) for k,v in m.items()}
        for i in range(a.paths):records.append(dict(policy=name,path=i,**{k:float(v[i]) for k,v in m.items()}))
        np.savetxt(out/f'{name}_trajectory.csv',tr,delimiter=',',comments='',
                   header='time,n1_q025,n2_q025,x_q025,n1_median,n2_median,x_median,n1_q975,n2_q975,x_q975,mean_u')
    with open(out/'path_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    paired={}
    if 'pinn' in allmetrics:
        for name,m in allmetrics.items():
            if name=='pinn':continue
            diff=allmetrics['pinn']['cost']-m['cost']
            se=diff.std(ddof=1)/np.sqrt(len(diff))
            paired[name]=dict(mean=float(diff.mean()),lower95=float(diff.mean()-1.96*se),
                              upper95=float(diff.mean()+1.96*se),method='normal approximation, paired paths')
    (out/'summary.json').write_text(json.dumps(dict(config=asdict(c),arguments=vars(a),
        metrics=summaries,paired_pinn_minus_baseline_cost=paired,
        note='Single training seed. Trajectory quantiles are not mean confidence intervals.'),indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

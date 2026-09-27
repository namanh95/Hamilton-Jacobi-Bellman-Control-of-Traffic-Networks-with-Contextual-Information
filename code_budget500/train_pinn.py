"""PINN trainer for the revised benchmark. Requires PyTorch.

An epoch is one pass through the fixed interior-plus-face training design.
Validation is not a uniform error certificate. Publication training is pending.
"""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.stats import qmc
import torch
from torch import nn
from model import coefficients, terminal, feedback, load_config

DTYPE = torch.float64


class ValueNet(nn.Module):
    def __init__(self,c,ansatz='soft'):
        super().__init__()
        self.config=c
        self.ansatz=ansatz
        self.register_buffer('upper',torch.tensor([c.K1,c.K2,1,c.T],dtype=DTYPE))
        layers=[]
        for nin,nout in [(4,64),(64,64),(64,64),(64,64),(64,1)]:
            layer=nn.Linear(nin,nout,dtype=DTYPE)
            nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
            layers.append(layer)
            if nout!=1:layers.append(nn.Tanh())
        self.net=nn.Sequential(*layers)
    def forward(self,p):
        raw=self.net(2*p/self.upper-1).squeeze(-1)
        if self.ansatz=='hard':
            return terminal(p[:,:3],self.config)+(self.config.T-p[:,3])*raw
        return raw


def sample(c,n,seed,kind='interior'):
    a=qmc.LatinHypercube(4,seed=seed).random(n)
    if kind=='terminal':a[:,3]=1
    if kind=='face':
        rng=np.random.default_rng(seed+987)
        ax=rng.integers(0,3,n); side=rng.integers(0,2,n)
        a[np.arange(n),ax]=side
    return a*np.array([c.K1,c.K2,1,c.T])


def residual(net,points,c):
    p=points.detach().clone().requires_grad_(True)
    v=net(p)
    grad=torch.autograd.grad(v.sum(),p,create_graph=True)[0]
    diag=[torch.autograd.grad(grad[:,i].sum(),p,create_graph=True)[0][:,i]
          for i in range(3)]
    u=feedback(grad,p[:,:3],c,torch)
    f,s,cost,_=coefficients(p[:,:3],p[:,3],u,c,torch)
    r=grad[:,3]+cost
    for i in range(3):r=r+grad[:,i]*f[i]+0.5*s[i]**2*diag[i]
    return r


def evaluate(net,c,interior,faces,terminal_points,batch=512):
    def stats(points,kind):
        total=0.; maximum=0.;count=0
        for q in points.split(batch):
            with torch.enable_grad():
                e=(residual(net,q,c) if kind=='residual'
                   else net(q)-terminal(q[:,:3],c))
            e=e.detach();total+=float((e*e).sum());count+=len(e)
            maximum=max(maximum,float(e.abs().max()))
        return total/count,maximum
    a,amax=stats(interior,'residual');b,bmax=stats(faces,'residual')
    d,dmax=stats(terminal_points,'terminal')
    return dict(interior_mse=a,face_mse=b,terminal_mse=d,
                interior_sample_max=amax,face_sample_max=bmax,
                terminal_sample_max=dmax,total=(2*a+b)/3+10*d)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--seed',type=int,default=11)
    p.add_argument('--epochs',type=int,default=20000)
    p.add_argument('--n-interior',type=int,default=20000)
    p.add_argument('--n-terminal',type=int,default=2000)
    p.add_argument('--n-validation',type=int,default=5000)
    p.add_argument('--n-test',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--validation-every',type=int,default=100)
    p.add_argument('--lbfgs-steps',type=int,default=100)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--output',default='results/seed11')
    a=p.parse_args()
    if min(a.epochs,a.n_interior,a.n_terminal,a.n_validation,a.n_test,a.batch_size,
           a.validation_every)<1:raise ValueError('Counts must be positive')
    c=load_config(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    net=ValueNet(c).to(a.device)
    def points(n,offset,kind='interior'):
        return torch.tensor(sample(c,n,a.seed+offset,kind),dtype=DTYPE,device=a.device)
    interior=points(a.n_interior,0)
    faces=points(max(1,a.n_interior//2),1,'face')
    train=torch.cat([interior,faces]);term=points(a.n_terminal,2,'terminal')
    val=(points(a.n_validation,10),points(max(1,a.n_validation//2),11,'face'),
         points(max(1,a.n_terminal//2),12,'terminal'))
    test=(points(a.n_test,20),points(max(1,a.n_test//2),21,'face'),
          points(a.n_terminal,22,'terminal'))
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    rng=np.random.default_rng(a.seed);best=float('inf');history=[];updates=0
    start=time.perf_counter()
    def snapshot(epoch,phase):
        nonlocal best
        metrics=evaluate(net,c,*val)
        row=dict(epoch=epoch,phase=phase,optimizer_steps=updates,
                 seconds=time.perf_counter()-start,**metrics)
        history.append(row)
        print(json.dumps(row),flush=True)
        with open(out/'training_history.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        if metrics['total']<best:
            best=metrics['total']
            torch.save(dict(state=net.state_dict(),config=asdict(c),epoch=epoch,
                            phase=phase,validation=metrics),out/'best.pt')
    snapshot(0,'initial')
    for epoch in range(1,a.epochs+1):
        order=rng.permutation(len(train))
        for k in range(0,len(order),a.batch_size):
            q=train[order[k:k+a.batch_size]]
            tq=term[rng.integers(0,len(term),min(a.batch_size,len(term)))]
            opt.zero_grad(set_to_none=True)
            loss=residual(net,q,c).square().mean()+10*(net(tq)-terminal(tq[:,:3],c)).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10)
            opt.step();updates+=1
        if epoch==1 or epoch%a.validation_every==0 or epoch==a.epochs:snapshot(epoch,'adam')
        if epoch in [2000,5000,10000,20000]:
            torch.save(dict(state=net.state_dict(),config=asdict(c),epoch=epoch),out/f'epoch_{epoch}.pt')
    if a.lbfgs_steps>0:
        saved=torch.load(out/'best.pt',map_location=a.device,weights_only=False)
        net.load_state_dict(saved['state'])
        lb=torch.optim.LBFGS(net.parameters(),max_iter=1,history_size=20,line_search_fn='strong_wolfe')
        closures=0
        def closure():
            nonlocal closures
            closures+=1;lb.zero_grad(set_to_none=True)
            objective=0.0
            for q in train.split(a.batch_size):
                loss=residual(net,q,c).square().sum()/len(train)
                loss.backward();objective+=float(loss.detach())
            for q in term.split(a.batch_size):
                loss=10*(net(q)-terminal(q[:,:3],c)).square().sum()/len(term)
                loss.backward();objective+=float(loss.detach())
            return torch.tensor(objective,dtype=DTYPE,device=a.device)
        for i in range(a.lbfgs_steps):
            lb.step(closure);updates+=1
            if (i+1)%10==0 or i+1==a.lbfgs_steps:snapshot(a.epochs+i+1,'lbfgs')
        (out/'lbfgs_evaluations.json').write_text(json.dumps(dict(closure_evaluations=closures)))
    saved=torch.load(out/'best.pt',map_location=a.device,weights_only=False)
    net.load_state_dict(saved['state'])
    report=evaluate(net,c,*test)
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2))
    (out/'run_metadata.json').write_text(json.dumps(dict(config=asdict(c),arguments=vars(a),
        torch_version=torch.__version__,numpy_version=np.__version__,seconds=time.perf_counter()-start,
        selected_epoch=saved['epoch'],selected_phase=saved['phase'],uniform_certificate=False),indent=2))


if __name__=='__main__':main()

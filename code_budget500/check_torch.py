"""Manufactured-solution and backend checks. Run before training on your machine."""
import json
import numpy as np
import torch
from torch import nn
from model import Config, coefficients, feedback
from train_pinn import residual,DTYPE


def main():
    c=Config();rng=np.random.default_rng(717)
    p=rng.uniform(.05,.95,(64,4))*np.array([c.K1,c.K2,1,c.T])
    class Manufactured(nn.Module):
        def forward(self,q):
            return (q[:,0]/c.K1)**2+2*(q[:,1]/c.K2)**2+3*q[:,2]**2+q[:,3]/c.T
    q=torch.tensor(p,dtype=DTYPE)
    grad=np.column_stack([2*p[:,0]/c.K1**2,4*p[:,1]/c.K2**2,6*p[:,2]])
    u=feedback(grad,p[:,:3],c)
    fs,ss,cost,_=coefficients(p[:,:3],p[:,3],u,c)
    expected=1/c.T+cost
    for i,second in enumerate([2/c.K1**2,4/c.K2**2,6]):
        expected=expected+grad[:,i]*fs[i]+.5*ss[i]**2*second
    actual=residual(Manufactured(),q,c).detach().numpy()
    error=float(np.max(np.abs(expected-actual)))
    assert np.allclose(expected,actual,rtol=1e-10,atol=1e-10)
    ft,st,ct,_=coefficients(q[:,:3],q[:,3],torch.tensor(u,dtype=DTYPE),c,torch)
    for a,b in zip(fs+ss+(cost,),ft+st+(ct,)):
        assert np.allclose(a,b.detach().numpy(),rtol=1e-12,atol=1e-12)
    print(json.dumps(dict(manufactured_residual_max_abs=error,
                         numpy_torch_coefficients='PASS'),indent=2))


if __name__=='__main__':main()

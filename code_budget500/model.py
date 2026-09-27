"""Revised benchmark shared by NumPy simulation and PyTorch training.

This is a new benchmark specification, not a reproduction of old paper tables.
States are physical [n1,n2,x]; time is minutes. Backend must be numpy or torch.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass(frozen=True)
class Config:
    K1: float = 4000.0
    K2: float = 4000.0
    T: float = 60.0
    beta1: float = 0.055
    beta2: float = 0.050
    alpha: float = 0.6
    gamma: float = 0.55
    kappa: float = 0.35
    eta: float = 0.16
    sigma1: float = 70.0
    sigma2: float = 70.0
    w1: float = 1.0
    w2: float = 0.7
    ru: float = 0.2
    bj: float = 0.2
    lost_weight: float = 5.0
    terminal_weight: float = 1.0
    demand_ref: float = 55.0
    demand_multiplier: float = 1.0


def load_config(path=None):
    c = Config(**json.load(open(path))) if path else Config()
    assert c.K1 > 0 and c.K2 > 0 and c.T > 0
    assert 0 < c.alpha < 1 and 0 <= c.gamma < 1 and c.ru > 0
    assert c.demand_multiplier >= 0 and c.demand_ref > 0
    assert min(c.beta1,c.beta2,c.kappa,c.eta,c.sigma1,c.sigma2,
               c.w1,c.w2,c.bj,c.lost_weight,c.terminal_weight) >= 0
    return c


def coefficients(z, t, u, c, xp=np):
    """Return drift/diffusion tuples, stage cost, and accounting flows."""
    n1, n2, x = z[..., 0], z[..., 1], z[..., 2]
    y1, y2 = n1 / c.K1, n2 / c.K2
    p1 = c.beta1 * n1 * (1-y1)
    p2 = c.beta2 * n2 * (1-y2)
    g1 = (1-c.gamma*x)*p1
    g2 = (1-c.alpha)*p2
    transfer = c.alpha*p2*(1-y1)
    demand = c.demand_multiplier*(55+35*xp.exp(-0.5*((t-24)/8)**2))
    mean = 0.05+0.85*xp.exp(-0.5*((t-30)/7)**2)
    admitted, rejected = demand*(1-y2), demand*y2
    f = (u*transfer-g1, admitted-u*transfer-g2, c.kappa*(mean-x))
    s = (c.sigma1*y1*(1-y1)*(0.25+0.75*x),
         c.sigma2*y2*(1-y2)*(0.25+0.75*x), c.eta*x*(1-x))
    cost = (c.w1*y1**2+c.w2*y2**2+c.ru*u**2+c.bj*(y1**8+y2**8)
            +c.lost_weight*rejected/c.demand_ref)
    flows = dict(offered=demand, admitted=admitted, rejected=rejected,
                 completed=g1+g2, transfer=transfer)
    return f, s, cost, flows


def terminal(z, c):
    return c.terminal_weight*((z[...,0]/c.K1)**2+(z[...,1]/c.K2)**2)


def feedback(gradient, z, c, xp=np):
    A = c.alpha*c.beta2*z[...,1]*(1-z[...,1]/c.K2)*(1-z[...,0]/c.K1)
    raw = A*(gradient[...,1]-gradient[...,0])/(2*c.ru)
    return xp.clip(raw, 0.0, 1.0)


def save_config(c, path):
    with open(path, 'w') as f:
        json.dump(asdict(c), f, indent=2)

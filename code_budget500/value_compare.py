"""Value and control comparison to an approximate finite grid, on fresh points."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from model import Config,feedback
from train_pinn import ValueNet,DTYPE


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(3)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=Config(**saved['config'])
    net=ValueNet(c,saved.get('ansatz','soft'));net.load_state_dict(saved['state']);net.eval()
    with np.load(a.reference) as d:
        if json.loads(str(d['config_json']))!=saved['config']:raise ValueError('Reference model mismatch')
        vi=RegularGridInterpolator((d['times'],d['y1'],d['y2'],d['x']),d['values'])
        # Control is piecewise constant at the temporal nodes, interpolated below.
        ui=RegularGridInterpolator((d['times'][:-1],d['y1'],d['y2'],d['x']),d['policies'])
        tmax=float(d['times'][-2])
    rng=np.random.default_rng(91671);q=rng.uniform(size=(10000,4));q[:,0]*=tmax
    physical=np.column_stack([q[:,1]*c.K1,q[:,2]*c.K2,q[:,3],q[:,0]])
    predictions=[];controls=[]
    for batch in np.array_split(physical,20):
        pt=torch.tensor(batch,dtype=DTYPE,requires_grad=True);v=net(pt)
        grad=torch.autograd.grad(v.sum(),pt)[0]
        predictions.extend(v.detach().numpy());controls.extend(feedback(grad,pt[:,:3],c,torch).detach().numpy())
    target=vi(q);e=np.array(predictions)-target;ue=np.array(controls)-ui(q)
    report=dict(points=len(q),value_rmse=float(np.sqrt(np.mean(e*e))),value_relative_l2=float(np.linalg.norm(e)/np.linalg.norm(target)),
                value_sample_max=float(np.max(np.abs(e))),policy_rmse=float(np.sqrt(np.mean(ue*ue))),
                reference=a.reference,checkpoint=a.checkpoint,note='Approximate grid reference; not a uniform error bound.')
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

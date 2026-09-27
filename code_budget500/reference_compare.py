"""Common-point diagnostics separating spatial, temporal and action refinements."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def compare(paths):
    rng=np.random.default_rng(80217)
    # At t=0, comparisons do not rely on temporal interpolation.
    points=rng.uniform(size=(10000,3));values={};initial={}
    for name,path in paths.items():
        with np.load(path) as d:
            ip=RegularGridInterpolator((d['y1'],d['y2'],d['x']),d['values'][0])
            values[name]=ip(points)
            initial[name]=dict(nominal=float(ip([[.425,.375,.05]])[0]),
                               congested=float(ip([[.75,.75,.5]])[0]))
    report=dict(initial_values=initial,comparisons={},points=10000,time=0,
                caveat='Grid differences, not an exact-error certificate.')
    for name,v in values.items():
        if name=='base':continue
        e=v-values['base']
        report['comparisons'][name+'_minus_base']=dict(rmse=float(np.sqrt(np.mean(e*e))),
            max_abs=float(np.max(np.abs(e))),relative_l2=float(np.linalg.norm(e)/np.linalg.norm(v)))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    paths={name:root/f'reference_{name}.npz' for name in ['base','space','time','action','fine']}
    report=compare(paths);(root/'reference_refinement.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

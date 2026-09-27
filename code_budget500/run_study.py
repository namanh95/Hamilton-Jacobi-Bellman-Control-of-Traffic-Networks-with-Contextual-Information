"""Stage-by-stage publication study driver. Re-run commands after interruption.

Default budgets preserve the proposed full study; --epochs may be lowered for
an explicitly labeled pilot. Completion of execution is not convergence proof.
"""
import argparse,csv,json,subprocess,sys
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from model import load_config,save_config
from evaluate import simulate

CODE=Path(__file__).resolve().parent
INITIALS={'nominal':[1700,1500,.05],'congested':[3000,3000,.5]}


def call(script,args):
    subprocess.run([sys.executable,str(CODE/script),*map(str,args)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='full_study');p.add_argument('--config')
    p.add_argument('--task',choices=['checks','train','references','tune','compare','checkpoints','summary'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    p.add_argument('--epochs',type=int,default=20000);p.add_argument('--lbfgs-steps',type=int,default=100)
    p.add_argument('--max-seconds',type=float,default=3600)
    p.add_argument('--ansatz',choices=['soft','hard'],default='hard')
    p.add_argument('--paths',type=int,default=200)
    a=p.parse_args();root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    c=load_config(a.config);config=root/'model.json'
    if config.exists() and json.loads(config.read_text())!=asdict(c):
        raise ValueError('Existing study uses a different model')
    save_config(c,config)
    def trained(seed):return root/f'{a.ansatz}_seed{seed}'
    if a.task=='checks':
        call('check_model.py',[]);call('check_torch.py',[])
    if a.task=='train':
        for seed in a.seeds:
            out=trained(seed)
            call('train_resumable.py',['--config',config,'--seed',seed,'--epochs',a.epochs,
                '--lbfgs-steps',a.lbfgs_steps,'--ansatz',a.ansatz,'--device',a.device,
                '--max-seconds',a.max_seconds,'--output',out,'--resume'])
            if not (out/'DONE.json').exists():
                print(f'Time slice saved for seed {seed}. Re-run the same train command.');return
    if a.task=='references':
        designs=[('base',13,9,21,.05),('space',17,13,21,.05),
                 ('time',13,9,21,.025),('action',13,9,41,.05),('fine',17,13,41,.025)]
        for name,n,nx,nu,dt in designs:
            out=root/f'reference_{name}.npz'
            valid=False
            if out.exists():
                try:
                    with np.load(out) as d:valid=d['values'].shape[0]>1
                except (ValueError,OSError,EOFError):valid=False
            if not valid:call('reference.py',['--config',config,'--nodes',n,'--context-nodes',nx,
                '--controls',nu,'--dt',dt,'--output',out])
        call('reference_compare.py',['--root',root])
    if a.task=='tune':
        rows=[]
        for gain in [.5,1,2,4]:
            means=[]
            for initial in INITIALS.values():
                rule=lambda z,t:np.clip(1-gain*np.maximum(z[:,0]/c.K1-.5,0),0,1)
                m,_=simulate(c,rule,initial,a.paths,.05,5678)
                means.append(float(m['cost'].mean()))
            rows.append(dict(gain=gain,mean_validation_cost=float(np.mean(means)),scenario_costs=means))
        best=min(rows,key=lambda r:r['mean_validation_cost'])
        (root/'reactive_tuning.json').write_text(json.dumps(dict(selected_gain=best['gain'],
            seed=5678,paths=a.paths,dt=.05,criterion='Equal weight nominal and congested mean cost',candidates=rows),indent=2))
    if a.task in ['compare','checkpoints']:
        ref=root/'reference_fine.npz'
        if not ref.exists():raise FileNotFoundError('Run references first')
        gain=json.loads((root/'reactive_tuning.json').read_text())['selected_gain']
        for seed in a.seeds:
            checkpoints=[('best',trained(seed)/'best.pt')] if a.task=='compare' else [
                (f'epoch_{k}',trained(seed)/f'epoch_{k}.pt') for k in [200,500,1000,2000,5000,10000,20000] if k<=a.epochs]
            for label,checkpoint in checkpoints:
                if not checkpoint.exists():raise FileNotFoundError(checkpoint)
                for scenario,initial in INITIALS.items():
                    for dt in ([.1,.05,.025] if label=='best' else [.05]):
                        out=root/f'eval_{a.ansatz}_{seed}_{label}_{scenario}_{dt}'
                        if (out/'summary.json').exists():continue
                        call('evaluate.py',['--config',config,'--checkpoint',checkpoint,'--reference',ref,
                            '--gain',gain,'--paths',a.paths,'--seed',10000+seed,'--dt',dt,
                            '--initial',*initial,'--output',out])
                if label=='best':
                    call('value_compare.py',['--checkpoint',checkpoint,'--reference',ref,
                        '--output',trained(seed)/'reference_errors.json'])
                    call('plot_results.py',['--training',trained(seed)/'training_history.csv',
                        '--evaluation',root/f'eval_{a.ansatz}_{seed}_best_congested_0.05',
                        '--output',root/f'figures_seed{seed}'])
    if a.task=='summary':
        rows=[];missing=[]
        for scenario in INITIALS:
            for dt in [.1,.05,.025]:
                for baseline in ['open','fixed','reactive','grid']:
                    diffs=[]
                    for seed in a.seeds:
                        path=root/f'eval_{a.ansatz}_{seed}_best_{scenario}_{dt}'/'summary.json'
                        if not path.exists():missing.append(str(path));continue
                        data=json.loads(path.read_text());diffs.append(data['paired_pinn_minus_baseline_cost'][baseline]['mean'])
                    n=len(diffs)
                    if n:
                        mean=float(np.mean(diffs));half=(float(student_t.ppf(.975,n-1)*np.std(diffs,ddof=1)/np.sqrt(n)) if n>1 else None)
                        rows.append(dict(scenario=scenario,dt=dt,baseline=baseline,training_seeds=n,
                            mean_paired_cost_difference=mean,lower95=None if half is None else mean-half,
                            upper95=None if half is None else mean+half))
        if rows:
            with (root/'comparison_summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (root/'study_status.json').write_text(json.dumps(dict(missing=sorted(set(missing)),
            statistical_unit='Independent training seed mean; distinct simulation seed per training run',
            caveat='Execution completion is not evidence of convergence, novelty, or superiority.'),indent=2))
        print(json.dumps(dict(rows=len(rows),missing=len(set(missing)))))


if __name__=='__main__':main()

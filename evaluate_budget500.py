"""Evaluate each immutable checkpoint when it becomes available; two CPU workers."""
import os,sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
BASE=Path(__file__).resolve().parent;ROOT=BASE/'budget500';CODE=BASE/'code_budget500'
sys.path.insert(0,str(CODE))
import torch
from train_pinn import ValueNet,sample,evaluate,DTYPE
from model import load_config

def call(name,args,log):
    with open(log,'w') as f:subprocess.run([sys.executable,str(CODE/name),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)

def checkpoint(seed,epoch):
    c=load_config(ROOT/'model.json');out=ROOT/f'hard_seed{seed}';p=out/f'epoch_{epoch}.pt'
    gain=json.loads((ROOT/'reactive_tuning.json').read_text())['selected_gain']
    for scenario,initial in {'nominal':[1700,1500,.05],'congested':[3000,3000,.5]}.items():
        dest=ROOT/f'eval_hard_{seed}_epoch_{epoch}_{scenario}_0.05'
        call('evaluate.py',['--checkpoint',p,'--reference',ROOT/'reference_fine.npz','--gain',gain,'--paths',200,'--seed',10000+seed,'--dt',.05,'--initial',*initial,'--output',dest],ROOT/f'eval_{seed}_{epoch}_{scenario}.log')
    # Tests are read after selection and do not choose the best checkpoint.
    torch.set_num_threads(1)
    saved=torch.load(p,map_location='cpu',weights_only=False);net=ValueNet(c,'hard');net.load_state_dict(saved['state'])
    points=lambda n,offset,kind:torch.tensor(sample(c,n,seed+offset,kind),dtype=DTYPE)
    metrics=evaluate(net,c,points(10000,20,'interior'),points(5000,21,'face'),points(2000,22,'terminal'))
    (out/f'test_epoch_{epoch}.json').write_text(json.dumps(metrics,indent=2))
    print('Checkpoint assessed',seed,epoch,flush=True)

def best(seed):
    call('run_study.py',['--root',ROOT,'--task','compare','--epochs',500,'--lbfgs-steps',0,'--seeds',seed],ROOT/f'compare_seed{seed}.log')
    print('Selected policy assessed',seed,flush=True)

pending=[(s,k) for k in [200,500,'best'] for s in [11,22,33,44,55]];futures=[]
start=time.monotonic()
with ThreadPoolExecutor(max_workers=2) as pool:
    while pending:
        if time.monotonic()-start>7200:raise TimeoutError('Training/evaluation exceeded two-hour watchdog')
        for s,k in pending[:]:
            target=ROOT/f'hard_seed{s}'/('DONE.json' if k=='best' else f'epoch_{k}.pt')
            if target.exists():
                futures.append(pool.submit(best,s) if k=='best' else pool.submit(checkpoint,s,k));pending.remove((s,k))
        for f in futures:
            if f.done():f.result()
        if pending:time.sleep(2)
    for f in futures:f.result()
call('run_study.py',['--root',ROOT,'--task','summary','--epochs',500],ROOT/'summary.log')
(ROOT/'EVALUATION_DONE.json').write_text(json.dumps({'status':'complete','seconds':time.monotonic()-start}))
print('ALL_EVALUATION_COMPLETE',flush=True)

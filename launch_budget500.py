import subprocess,sys,time,json,os
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'budget500'
children=[]
for seed in [11,22,33,44,55]:
    log=(out/f'train_seed{seed}.log').open('w')
    cmd=[sys.executable,str(root/'code_budget500/train_resumable.py'),'--seed',str(seed),'--epochs','500','--lbfgs-steps','0','--ansatz','hard','--threads','1','--validation-every','25','--output',str(out/f'hard_seed{seed}'),'--resume']
    children.append((seed,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
    print('Started seed',seed,flush=True)
for seed,p,log in children:
    rc=p.wait();log.close();print('Finished seed',seed,'exit',rc,flush=True)
    if rc:raise RuntimeError(f'Seed {seed} failed; see log')
print('ALL_FIVE_COMPLETE',flush=True)

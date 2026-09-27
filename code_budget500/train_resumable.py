"""Resumable full-design PINN training with explicit soft/hard terminal ablation.

The hard ansatz is Phi(z)+(T-t)*N_theta(z,t), enforcing terminal data exactly.
It is an architectural change, not an artificially reported zero fitting error.
Checkpoint and logs are atomic. Resuming retains Adam, RNG and L-BFGS state.
"""
import argparse,csv,hashlib,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from train_pinn import ValueNet,sample,residual,evaluate,DTYPE
from model import load_config,terminal


def atomic_torch(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def atomic_json(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def run(a):
    c=load_config(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    rng=np.random.default_rng(a.seed);net=ValueNet(c,a.ansatz).to(a.device)
    def points(n,offset,kind='interior'):
        return torch.tensor(sample(c,n,a.seed+offset,kind),dtype=DTYPE,device=a.device)
    train=torch.cat([points(a.n_interior,0),points(max(1,a.n_interior//2),1,'face')])
    terms=points(a.n_terminal,2,'terminal')
    val=(points(a.n_validation,10),points(max(1,a.n_validation//2),11,'face'),
         points(max(1,a.n_terminal//2),12,'terminal'))
    test=(points(a.n_test,20),points(max(1,a.n_test//2),21,'face'),points(a.n_terminal,22,'terminal'))
    signature=dict(config=asdict(c),seed=a.seed,ansatz=a.ansatz,n_interior=a.n_interior,
        n_terminal=a.n_terminal,n_validation=a.n_validation,n_test=a.n_test,batch=a.batch_size)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    lb=torch.optim.LBFGS(net.parameters(),max_iter=1,history_size=20,line_search_fn='strong_wolfe')
    state=dict(phase='adam',epoch=0,lbfgs_step=0,steps=0,closures=0,seconds=0.,best=float('inf'),history=[])
    if (out/'resume.pt').exists():
        if not a.resume:raise FileExistsError('Use --resume or choose a new output directory')
        loaded=torch.load(out/'resume.pt',map_location=a.device,weights_only=False)
        if loaded['signature']!=signature:raise ValueError('Resume configuration mismatch')
        net.load_state_dict(loaded['state']);opt.load_state_dict(loaded['adam'])
        lb.load_state_dict(loaded['lbfgs']);state=loaded['progress']
        rng.bit_generator.state=loaded['numpy_rng']
        torch.set_rng_state(loaded['torch_rng'].cpu())
        if a.device=='cuda' and loaded.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in loaded['cuda_rng']])
        if state['phase']!='adam' and a.epochs>state['epoch']:
            raise ValueError('Cannot extend Adam after L-BFGS has begun; use a new run or keep the original target')
    elif a.resume:
        print('No checkpoint: starting new run',flush=True)
    base_seconds=state['seconds'];start=time.perf_counter()
    def save():
        state['seconds']=base_seconds+time.perf_counter()-start
        atomic_torch(dict(signature=signature,state=net.state_dict(),adam=opt.state_dict(),
            lbfgs=lb.state_dict(),progress=state,numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else None),out/'resume.pt')
    def snapshot(phase):
        metrics=evaluate(net,c,*val)
        row=dict(epoch=state['epoch'],phase=phase,optimizer_steps=state['steps'],
                 seconds=base_seconds+time.perf_counter()-start,**metrics)
        state['history'].append(row)
        tmp=out/'training_history.csv.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(state['history'])
        os.replace(tmp,out/'training_history.csv')
        if metrics['total']<state['best']:
            state['best']=metrics['total']
            atomic_torch(dict(state=net.state_dict(),config=asdict(c),ansatz=a.ansatz,
                epoch=state['epoch'],phase=phase,validation=metrics),out/'best.pt')
        print(json.dumps(row),flush=True)
    def expired():return a.max_seconds>0 and time.perf_counter()-start>=a.max_seconds
    if not state['history']:snapshot('initial');save()
    if state['phase']=='done':
        if a.lbfgs_steps>state['lbfgs_step']:
            raise ValueError('Completed run cannot extend refinement; choose a new run')
        print('Run already complete',flush=True);return
    if state['phase']=='adam':
        for epoch in range(state['epoch']+1,a.epochs+1):
            order=rng.permutation(len(train))
            for k in range(0,len(train),a.batch_size):
                q=train[order[k:k+a.batch_size]]
                tq=terms[rng.integers(0,len(terms),min(a.batch_size,len(terms)))]
                opt.zero_grad(set_to_none=True)
                loss=residual(net,q,c).square().mean()
                if a.ansatz=='soft':loss=loss+10*(net(tq)-terminal(tq[:,:3],c)).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
                state['steps']+=1
            state['epoch']=epoch
            if epoch==1 or epoch%a.validation_every==0 or epoch==a.epochs or expired():
                snapshot('adam');save()
            if epoch in [200,500,1000,2000,5000,10000,20000]:
                atomic_torch(dict(state=net.state_dict(),config=asdict(c),ansatz=a.ansatz,
                    epoch=epoch,phase='adam'),out/f'epoch_{epoch}.pt')
            if expired() and epoch<a.epochs:print('TIME_SLICE_COMPLETE: rerun with --resume',flush=True);return
        best=torch.load(out/'best.pt',map_location=a.device,weights_only=False)
        net.load_state_dict(best['state']);state['phase']='lbfgs';save()
    def closure():
        state['closures']+=1;lb.zero_grad(set_to_none=True);value=0.
        for q in train.split(a.batch_size):
            loss=residual(net,q,c).square().sum()/len(train);loss.backward();value+=float(loss.detach())
        if a.ansatz=='soft':
            for q in terms.split(a.batch_size):
                loss=10*(net(q)-terminal(q[:,:3],c)).square().sum()/len(terms)
                loss.backward();value+=float(loss.detach())
        if not np.isfinite(value):raise RuntimeError('Nonfinite L-BFGS objective')
        return torch.tensor(value,dtype=DTYPE,device=a.device)
    for k in range(state['lbfgs_step']+1,a.lbfgs_steps+1):
        lb.step(closure);state['steps']+=1;state['lbfgs_step']=k
        if k%10==0 or k==a.lbfgs_steps or expired():snapshot('lbfgs');save()
        if expired() and k<a.lbfgs_steps:print('TIME_SLICE_COMPLETE: rerun with --resume',flush=True);return
    best=torch.load(out/'best.pt',map_location=a.device,weights_only=False)
    net.load_state_dict(best['state']);report=evaluate(net,c,*test)
    atomic_json(report,out/'test_metrics.json')
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    atomic_json(dict(config=asdict(c),arguments=vars(a),torch_version=torch.__version__,
        numpy_version=np.__version__,seconds=base_seconds+time.perf_counter()-start,
        selected_epoch=best['epoch'],selected_phase=best['phase'],lbfgs_closures=state['closures'],
        source_sha256=source_hashes,uniform_certificate=False),out/'run_metadata.json')
    state['phase']='done';save();atomic_json(dict(status='COMPLETED',test=report),out/'DONE.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--seed',type=int,default=11)
    p.add_argument('--epochs',type=int,default=20000);p.add_argument('--lbfgs-steps',type=int,default=100)
    p.add_argument('--n-interior',type=int,default=20000);p.add_argument('--n-terminal',type=int,default=2000)
    p.add_argument('--n-validation',type=int,default=5000);p.add_argument('--n-test',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--validation-every',type=int,default=100)
    p.add_argument('--ansatz',choices=['soft','hard'],default='hard');p.add_argument('--resume',action='store_true')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--max-seconds',type=float,default=0);p.add_argument('--output',default='results/resumable_seed11')
    args=p.parse_args()
    if min(args.epochs,args.n_interior,args.n_terminal,args.n_validation,args.n_test,args.batch_size,args.validation_every,args.threads)<1:
        p.error('Counts must be positive')
    run(args)

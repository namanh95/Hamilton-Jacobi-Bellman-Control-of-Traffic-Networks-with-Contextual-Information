from pathlib import Path
import json,math,csv,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;R=B/'budget500';O=B/'budget500_report';O.mkdir(exist_ok=True)
seeds=[11,22,33,44,55]
def read(p):return json.loads(p.read_text())
def ci(x):
    x=np.asarray(x);m=x.mean();h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return float(m),float(m-h),float(m+h)
def table(df,formats=None):
    formats=formats or {};lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for _,r in df.iterrows():lines.append('| '+' | '.join(formats.get(k,lambda v:str(v))(v) for k,v in r.items())+' |')
    return '\n'.join(lines)
rows=[]
for s in seeds:
    d=R/f'hard_seed{s}';m=read(d/'run_metadata.json');v=read(d/'test_metrics.json');e=read(d/'reference_errors.json')
    rows.append(dict(seed=s,selected_epoch=m['selected_epoch'],interior_mse=v['interior_mse'],face_mse=v['face_mse'],terminal_mse=v['terminal_mse'],sample_max=max(v['interior_sample_max'],v['face_sample_max']),value_relative_l2=e['value_relative_l2'],policy_rmse=e['policy_rmse'],training_seconds=m['seconds']))
acc=pd.DataFrame(rows);acc.to_csv(O/'accuracy.csv',index=False)
check=[]
for s in seeds:
    d=R/f'hard_seed{s}'
    for k in [200,500]:
        e=read(d/f'test_epoch_{k}.json');check.append(dict(seed=s,epoch=k,interior_mse=e['interior_mse'],face_mse=e['face_mse'],weighted_mse=e['total']))
ck=pd.DataFrame(check);ck.to_csv(O/'checkpoint_accuracy.csv',index=False)
policy=[];changes=[];operational=[]
for scenario in ['nominal','congested']:
    for dt in [.1,.05,.025]:
        ds=[read(R/f'eval_hard_{s}_best_{scenario}_{dt}/summary.json') for s in seeds]
        for name in ['open','fixed','reactive','pinn','grid']:
            m,l,u=ci([d['metrics'][name]['cost']['mean'] for d in ds]);policy.append(dict(scenario=scenario,dt=dt,policy=name,mean_cost=m,lower95=l,upper95=u))
        if dt==.05:
            for name in ['open','fixed','reactive','pinn','grid']:
                operational.append(dict(scenario=scenario,policy=name,**{k:np.mean([d['metrics'][name][k]['mean'] for d in ds]) for k in ['vehicle_hours','completed','rejected','saturation','projection_vehicle_abs','projection_context_abs','balance_error_max']}))
    for s in seeds:
        d200=read(R/f'eval_hard_{s}_epoch_200_{scenario}_0.05/summary.json');d500=read(R/f'eval_hard_{s}_epoch_500_{scenario}_0.05/summary.json')
        changes.append(dict(seed=s,scenario=scenario,cost_epoch200=d200['metrics']['pinn']['cost']['mean'],cost_epoch500=d500['metrics']['pinn']['cost']['mean'],difference_500_minus_200=d500['metrics']['pinn']['cost']['mean']-d200['metrics']['pinn']['cost']['mean']))
pol=pd.DataFrame(policy);pol.to_csv(O/'policy_costs.csv',index=False)
chg=pd.DataFrame(changes);chg.to_csv(O/'checkpoint_policy_changes.csv',index=False)
pd.DataFrame(operational).to_csv(O/'operational_metrics.csv',index=False)
paired=pd.read_csv(R/'comparison_summary.csv');paired.to_csv(O/'paired_comparisons.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
for s in seeds:
    h=pd.read_csv(R/f'hard_seed{s}/training_history.csv');ax.semilogy(h.epoch,(2*h.interior_mse+h.face_mse)/3,label=f'Seed {s}')
ax.axvline(200,color='grey',linestyle=':',linewidth=1);ax.set(xlabel='Adam epochs (30 updates per epoch)',ylabel='Held-out validation residual MSE',title='Five independent 500-epoch runs');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'five_seed_convergence.png',dpi=180);fig.savefig(O/'five_seed_convergence.pdf');plt.close(fig)
q=pol[pol.dt==.05].pivot(index='policy',columns='scenario',values='mean_cost').reset_index()
report='''# Five-seed 500-epoch study: measured results

**Status:** all five 500-epoch Adam runs and the specified comparisons completed. This is a finite-budget numerical assessment, not proof of PINN convergence or a submission-ready manuscript. These experiments use the corrected synthetic benchmark; they do not reproduce the old manuscript's numerical tables.

## Design

Seeds 11, 22, 33, 44 and 55; 500 Adam epochs per seed; 30 updates per epoch; 75,000 Adam updates in total. Each epoch covers 20,000 interior and 10,000 boundary-face points. Architecture: four hidden layers of 64 tanh units, float64. The model and architecture were held fixed. Terminal values are imposed by J=Phi+(T-t)N, so zero terminal error is by construction. Validation occurs every 25 epochs, and its minimum weighted residual selects the final policy. Independent test sets do not select models. L-BFGS is omitted because neither earlier pilot improved under that refinement. Actual 200- and 500-epoch checkpoints are assessed separately from the selected checkpoint.

The five runs used concurrent one-thread CPU workers. Their per-run times therefore differ from the earlier sequential estimates. Summing these elapsed times is not the total wall-clock duration. Numerical evaluation uses 200 paired paths per controller, two initial states, and three time steps. The simulation seed is 10000 plus the training seed. Intervals below use the five independent training-and-simulation replication means (Student t, 4 degrees of freedom); they combine training and simulation variability, with limited precision at n=5.

## Selected-checkpoint test accuracy

'''
report+=table(acc.rename(columns={'value_relative_l2':'value_relative_L2'}),{k:(lambda v:f'{v:.6g}') for k in acc.columns if k not in ['seed','selected_epoch']})+'\n\n'
report+='The grid value and policy comparisons use 10,000 common fresh space-time points. The reference is approximate; relative L2 discrepancy is not error against an exact solution. Sampled maxima are not uniform bounds.\n\n## Mean objective costs at dt=0.05 minutes\n\n'+table(q,{k:(lambda v:f'{v:.6f}') for k in ['nominal','congested']})+'\n\n'
report+='## Paired cost differences: PINN minus baseline\n\nNegative values favor the PINN under this objective.\n\n'+table(paired[paired.dt==.05],{k:(lambda v:f'{v:.6g}') for k in ['mean_paired_cost_difference','lower95','upper95']})+'\n\n'
report+='## Changes between actual 200- and 500-epoch checkpoints\n\n'
for scenario in ['nominal','congested']:
    m,l,u=ci(chg[chg.scenario==scenario].difference_500_minus_200)
    report+=f'- {scenario}: mean cost change {m:.6g}, across-replication 95% interval [{l:.6g}, {u:.6g}].\n'
for k in [200,500]:
    d=ck[ck.epoch==k];report+=f'- Epoch {k}: test weighted residual MSE mean {d.weighted_mse.mean():.6g}, range [{d.weighted_mse.min():.6g}, {d.weighted_mse.max():.6g}].\n'
wide=ck.pivot(index='seed',columns='epoch',values='weighted_mse')
n_worse=int((wide[500]>wide[200]).sum())
report+=f'\n**Endpoint check:** {n_worse} of five runs have a higher weighted test residual at epoch 500 than at epoch 200. Best-validation selection and final-epoch behavior are reported separately; the selected-checkpoint accuracy does not establish monotone convergence.\n'
report+='''
Inspect the full curves and checkpoint table rather than interpreting a small policy-cost change as proof of PDE convergence. The final epoch need not be the best validated checkpoint. This study deliberately makes no claim of asymptotic stabilization.

## Numerical-reference and simulation checks

The reference comparisons separate spatial (13x13x9 versus 17x17x13), time (0.05 versus 0.025 minutes) and action (21 versus 41) refinements. The final comparison uses 17x17x13 nodes, 41 actions and dt=0.025. Spatial changes remain material: approximately 0.46% relative L2 between the coarse and fine designs at t=0. These are refinement differences, not a certified error estimate. Slightly lower simulated cost than the grid policy is not evidence of beating the continuous optimal controller.

Simulation results at dt=0.1, 0.05 and 0.025 are in policy_costs.csv and paired_comparisons.csv. Controllers share random increments within each scenario/time-step comparison. Brownian paths are not nested across time-step sizes; cross-step changes are empirical mean comparisons, not a pathwise convergence experiment. Baseline gain selection used a separate seed 5678 and the equal-weight average of the two scenarios, selecting gain 0.5 from 0.5,1,2,4. The gain is on the candidate boundary, so this is a tuned candidate-set baseline, not proof of globally optimal reactive tuning.

Operational metrics include completed trips, rejected demand, vehicle-hours, saturation and numerical projection corrections. Demand rejected by the admission rule is lost/diverted, not queued upstream. Objective reductions must not be equated with system-wide travel-time reductions. In both tested scenarios, the selected PINN completes fewer trips and rejects more demand than fully open control. The nominal means are about 3303.81 versus 3325.84 completed trips, and 1605.28 versus 1593.94 rejected vehicles. The congested means are about 3505.18 versus 3531.03 completions and 2825.26 versus 2808.29 rejections. Thus, the objective improvement does not demonstrate better throughput or reduced demand loss; the objective also penalizes control effort.

## Reviewer-response implications

The numerical portion now includes actual multi-seed evidence, independent test diagnostics, both uncongested and above-critical initial states, a tuned reactive baseline, open/fixed/grid comparisons, and refinement checks. Algorithm 1 and Figure 1 must be synchronized with exact terminal enforcement. The convergence figure must show the actual budget and optimizer-update count. The older manuscript's 2,000-epoch convention is not established to match this implementation's epoch definition.

Reviewer #2's request for longer training is addressed through a revised architecture and stronger finite-budget accuracy evidence; it is not literally fulfilled by training beyond 2,000 epochs. If residuals continue changing, acknowledge that limitation directly. Acceptance cannot be inferred from these experiments. Publication-level novelty remains a separate assessment.

Training beyond 500 epochs, broader parameter robustness, real-city calibration, observed-context errors, upstream queues, larger networks and long-run guarantees remain outside this study. Do not claim these were validated. No automatic extension to 1,000 epochs was performed.
'''
(O/'RESULTS.md').write_text(report)
# Standalone LaTeX numerical note, with honest scope and directly generated tables.
def tex_table(df,caption):
    def fmt(x):
        return str(x).replace('_',r'\_') if isinstance(x,str) else f'{x:.6g}'
    rows=[' & '.join(map(fmt,df.columns))+r' \\ \midrule']
    rows += [' & '.join(fmt(v) for v in row)+r' \\' for row in df.itertuples(index=False,name=None)]
    return (r'\begin{table}[H]\centering\small\caption{'+caption+'}'+
            r'\begin{tabular}{'+'l'*len(df.columns)+r'}\toprule '+ '\n'.join(rows)+
            r'\bottomrule\end{tabular}\end{table}'+'\n')
tex=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,graphicx,amsmath,float}
\title{Finite-budget PINN validation: five independent 500-epoch runs}
\author{Numerical revision working note}
\date{}
\begin{document}\maketitle
This note reports newly computed results for the corrected synthetic benchmark. It is not a claim of asymptotic convergence. Each of five independent initializations is trained for 500 Adam epochs, with 30 minibatch updates per epoch. The exact-terminal construction $J_\theta(z,t)=\Phi(z)+(T-t)N_\theta(z,t)$ imposes the terminal condition. Consequently, zero terminal error is an architectural property, not an observed convergence result. Selection uses validation residuals; independent test data do not select models. L-BFGS is omitted after unsuccessful pilot refinements.
'''
tex+=tex_table(acc[['seed','selected_epoch','interior_mse','face_mse','value_relative_l2']].rename(columns={'selected_epoch':'Selected epoch','interior_mse':'Interior MSE','face_mse':'Face MSE','value_relative_l2':'Value relative L2'}),'Selected-checkpoint diagnostics; the value comparison uses an approximate grid reference.')
tex+=tex_table(q,'Mean objective costs over five training-and-simulation replications; 200 paired paths per replication, time step 0.05 minutes. Lower is better.')
tex+=r'''\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{five_seed_convergence.png}\caption{Actual held-out validation residuals. All five initialization seeds are shown. Terminal error is exactly zero by construction.}\end{figure}
The epoch-500 weighted test residual exceeds its epoch-200 value for one of the five seeds. Four selected policies come from epoch 475, and one from epoch 500. Thus, best-checkpoint accuracy must not be interpreted as established training convergence.

Despite lower objective costs, the PINN completes fewer trips and rejects more demand than fully open control in both tested scenarios. The objective includes a control-effort penalty, so these results do not demonstrate improved traffic throughput.

The grid reference uses $17\times17\times13$ nodes and 41 actions. Spatial refinement changes the value materially, so grid agreement cannot certify exact optimality. The study includes nominal and congested initial conditions, an independently tuned reactive candidate, and time-step checks at 0.1, 0.05 and 0.025 minutes. Parameter stress sweeps and real-world calibration are not included. The training budget is smaller than the duration requested by Reviewer 2; the revised evidence supports only the measured finite-budget accuracy and policy performance. Longer-budget convergence remains unestablished. Full per-seed outputs, confidence intervals, checkpoint changes and accounting metrics accompany this note.
\end{document}
'''
(O/'numerical_note.tex').write_text(tex)
print(report[:500]);print('Reports written',O)

"""Plot actual run logs; no synthetic or smoothed convergence curves."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',required=True,help='training_history.csv')
    p.add_argument('--evaluation',required=True,help='evaluation directory')
    p.add_argument('--output',default='figures/generated');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    h=pd.read_csv(a.training)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name,label in [('interior_mse','Interior residual'),('face_mse','Face residual')]:
        axes[0].semilogy(h.optimizer_steps,h[name],label=label)
    if (h.terminal_mse == 0).all():
        axes[1].plot(h.optimizer_steps,h.terminal_mse,label='Exactly imposed terminal condition')
        axes[1].text(.5,.7,'Zero by construction; not a convergence result',
                     ha='center',transform=axes[1].transAxes,fontsize=8)
    else:
        axes[1].semilogy(h.optimizer_steps,h.terminal_mse,label='Terminal condition')
    for ax in axes:
        ax.set(xlabel='Optimizer steps',ylabel='Validation mean squared error')
        ax.legend();ax.grid(alpha=.2)
    lb=h[h.phase=='lbfgs']
    if len(lb):
        for ax in axes:ax.axvline(lb.optimizer_steps.iloc[0],ls='--',c='grey')
    fig.tight_layout();fig.savefig(out/'convergence.pdf');fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for path in sorted(Path(a.evaluation).glob('*_trajectory.csv')):
        d=pd.read_csv(path);name=path.name.replace('_trajectory.csv','')
        axes[0].plot(d.time,d.n1_median,label=name)
        axes[0].fill_between(d.time,d.n1_q025,d.n1_q975,alpha=.12)
        axes[1].plot(d.time,d.mean_u,label=name)
    axes[0].set(ylabel='Core accumulation (vehicles)',title='Median and empirical 95% trajectory interval')
    axes[1].set(xlabel='Time (minutes)',ylabel='Mean transfer fraction')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'trajectories.pdf');fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

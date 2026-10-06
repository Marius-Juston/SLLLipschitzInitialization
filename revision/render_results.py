"""Generate manuscript tables and a diagnostic figure from audited results."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/campaign')


def render():
    summary=json.loads((ROOT/'validated-summary.json').read_text())
    groups={(g['dataset'],g['model'],g['depth'],g['bias']):g['metrics'] for g in summary['groups']}
    def fmt(group,metric):
        m=groups[group][metric]
        return f"{100*m['mean']:.2f} $\\pm$ {100*m['sd']:.2f}"
    out=[]
    def table(caption,columns,header,rows,label,wide=True):
        env = 'table*' if wide else 'table'
        out.extend([r'\begin{'+env+r'}[pos=htbp]',r'\centering',r'\small',
                    *([] if wide else [r'\setlength{\tabcolsep}{3pt}']),r'\caption{'+caption+'}',
                    r'\begin{tabular}{'+columns+'}',header+r' \\',r'\hline',*rows,
                    r'\end{tabular}',r'\label{'+label+'}',r'\end{'+env+'}',''])
    rows=[]
    for model in ('aol','sll'):
        for depth in (5,15,30):
            zero=('cifar10',model,depth,'zero');corr=(*zero[:3],'corrected')
            rows.append(f"{model.upper()} & {depth} & {fmt(zero,'clean')} & {fmt(corr,'clean')} & {fmt(zero,'certified_0.5')} & {fmt(corr,'certified_0.5')} \\\\")
    table('Feedforward CIFAR-10 accuracy (percent; mean $\\pm$ sample standard deviation over three seeds). Certification uses $\\ell_2$ radius $0.5$ on all 10,000 test images. Depth counts hidden layers; each model has an additional linear logit head. Corrected denotes the variance-based bias heuristic.',
          'llrrrr','Model & Depth & Clean: zero & Clean: corrected & Certified: zero & Certified: corrected',rows,'tab:feedforward-results')
    rows=[]
    for depth in (5,15,30):
        rows.append(str(depth)+' & '+' & '.join(fmt(('covertype','sll',depth,b),'clean') for b in ('zero','default','corrected'))+r' \\')
    table('Covertype test accuracy (percent; mean $\\pm$ sample standard deviation over ten seeds). All models use the same 116,203 test examples and 25-epoch training budget.',
          'lrrr','Depth & Zero & Default & Corrected',rows,'tab:covertype-results',wide=False)
    rows=[]
    for metric,label in [('clean','Clean'),('noise_0.01',r'Noise $s=0.01$'),
                         ('noise_0.03',r'Noise $s=0.03$'),('noise_0.05',r'Noise $s=0.05$')]:
        rows.append(label+' & '+' & '.join(fmt(('cifar10','residual',20,bias),metric)
                    for bias in ('default','corrected'))+r' \\')
    table('Residual SLL CIFAR-10 clean and noisy-input accuracy (percent; mean $\\pm$ sample standard deviation over three seeds, all 10,000 test images). Columns compare default and corrected biases. Gaussian noise is clipped to $[0,1]$; $s$ denotes its standard deviation before clipping.',
          'lrr','Evaluation & Default & Corrected',rows,'tab:residual-noise',wide=False)
    rows=[]
    for eps in ('0.25','0.5','1.0'):
        for bias in ('default','corrected'):
            g=('cifar10','residual',20,bias)
            rows.append(eps+' & '+bias.capitalize()+' & '+' & '.join(fmt(g,m+'_'+eps) for m in ('certified','subset_certified','pgd','autoattack'))+r' \\')
    table('Residual SLL CIFAR-10 certified and attack accuracy (percent; mean $\\pm$ sample standard deviation over three seeds). Full-test certificates use 10,000 examples; subset certificates, PGD and AutoAttack use the same fixed 1,000 examples. PGD uses 100 steps and five restarts; AutoAttack uses its standard $\\ell_2$ configuration.',
          'llrrrr',r'Radius & Bias & Certified: full & Certified: subset & PGD: subset & AutoAttack: subset',rows,'tab:residual-robustness')
    (ROOT/'result-tables.tex').write_text('\n'.join(out))
    records=json.loads((ROOT/'precision-diagnostics.json').read_text())['records']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5),sharex=True,layout='constrained')
    exported={}
    for bias,color in [('zero','#225ea8'),('corrected','#d95f0e')]:
        for axis,metric,title in zip(axes,('second_moment','input_variance'),('Activation second moment','Variance across inputs')):
            values=np.array([[records[f'cifar10-sll-d15-{bias}-s{seed}'][f'net.{i}'][metric]
                              for i in range(1,16)] for seed in range(3)])
            exported[bias+'_'+metric]=values.tolist()
            floor=1e-30; x=np.arange(1,16)
            axis.fill_between(x,np.maximum(values.min(0),floor),np.maximum(values.max(0),floor),color=color,alpha=.15)
            axis.plot(x,np.maximum(values.mean(0),floor),label=bias.capitalize(),color=color,marker='o',markersize=2)
            axis.set(yscale='log',title=title,xlabel='Hidden layer',xticks=[1,5,10,15]);axis.grid(alpha=.2)
    axes[0].legend(frameon=False);axes[1].set_ylim(3e-31,1e-1)
    fig.savefig(ROOT/'initialization-diagnostics.pdf');fig.savefig(ROOT/'initialization-diagnostics.png',dpi=180)
    (ROOT/'plotted-diagnostics.json').write_text(json.dumps(exported,indent=2)+'\n')
    print(ROOT/'result-tables.tex')


if __name__=='__main__':render()

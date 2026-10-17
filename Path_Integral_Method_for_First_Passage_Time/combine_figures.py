from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
})

fig = plt.figure(figsize=(7.2, 10), constrained_layout=True)
rows = fig.subfigures(4, 1)

for i, row in enumerate(rows):
    row.suptitle(f'Example {i+1}', fontsize=11, fontweight='bold')
    left, right = row.subplots(1, 2)

    # 读取已有结果，不重新仿真。
    folder = base / f'fpt_results_E{i+1}'
    t, rho, H, H_mc = np.loadtxt(
        folder / 'curves.csv',
        delimiter=',', skiprows=1, unpack=True
    )
    T = np.load(folder / 'first_passage_times.npy')
    edges = np.linspace(0, t[-1], 101)
    counts, _ = np.histogram(T[np.isfinite(T)], edges)

    # 左列：理论密度和仿真直方图。
    left.plot(
        t[1:], rho[1:], color='#b33440',
        lw=1.5, label='Theory', zorder=3
    )
    left.bar(
        (edges[:-1]+edges[1:])/2,
        counts/(T.size*np.diff(edges)),
        width=np.diff(edges), color='#bed5e8',
        edgecolor='none', label='Simulation', zorder=1
    )

    # 右列：理论实线在下层，仿真虚线在上层。
    right.plot(
        t, H, color='#b33440',
        lw=1.5, label='Theory', zorder=2
    )
    right.plot(
        t, H_mc, color='#24658b', ls='--',
        lw=1.5, label='Simulation', zorder=3
    )

    left.set_ylabel(r'$\rho(t)$')
    right.set_ylabel(r'$H(t)$')
    left.set_ylim(bottom=0)
    right.set_ylim(0, 1)

    for j, (ax, title) in enumerate(zip(
        (left, right), ('FPT Density', 'Survival probability')
    )):
        ax.set_title(
            f'({chr(97+2*i+j)}) {title}',
            loc='left', fontsize=9
        )
        ax.set(xlabel=r'Time $t$', xlim=(0, t[-1]))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out')

        # 只有第一行的两个子图显示图例。
        if i == 0:
            ax.legend(frameon=False, loc='upper right')

for extension in ('pdf', 'png', 'svg'):
    output = base / f'comparison_all.{extension}'
    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved: {output}')

plt.close(fig)
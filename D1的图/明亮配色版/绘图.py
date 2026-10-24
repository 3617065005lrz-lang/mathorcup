"""图2—图4明亮配色版：保留真实统计，SVG文字转轮廓。"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

HERE=Path(__file__).resolve().parent
CACHE=HERE.parents[1]/'第一问/03_可视化/plot_cache'
EVENT_COLOR='#29A8E0'  # 事件组：明亮天蓝
FACTOR_COLORS={'土壤湿润状态':'#29C7A5','热量累积':'#FF805C','辐射能量':'#F5BE3D','降水与水分输入':'#29A8E0','大气湿度与水汽':'#A38BE4'}
HISTORY_COLOR='#FF805C'  # 历史组：珊瑚红
INK='#202020'
MUTED='#484848'
PALE='#D4D4D4'
WIDTH=17.5/2.54
STYLE={'font.family':['Arial','Microsoft YaHei','sans-serif'],'axes.unicode_minus':False,
       'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.5,
       'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
       'axes.edgecolor':'#666666','axes.linewidth':.6,'savefig.bbox':None,
       'svg.fonttype':'path','figure.facecolor':'white','axes.facecolor':'white'}


def load():
    meta=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
    assert meta['control_years']==[2021,2022,2023,2024] and meta['excluded_validation_year']==2025
    return {'stability':pd.read_csv(CACHE/'factor_stability.csv'),
            'top5':pd.read_csv(CACHE/'top5_factor_metadata.csv'),
            'grid':pd.read_csv(CACHE/'top5_sensitivity_grid.csv'),
            'distribution':pd.read_parquet(CACHE/'top5_event_control_distribution.parquet')}


def polish(ax,grid='y'):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5,width=.6)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color='#E8ECEF',lw=.5)


def header(fig,title):
    fig.text(.5,.968,title,ha='center',va='top',fontsize=11,fontweight='normal')


def save(fig,name):
    fig.savefig(HERE/(name+'.png'),dpi=600,facecolor='white',bbox_inches=None)
    for ax in fig.axes:
        for col in ax.collections:col.set_rasterized(False)
    fig.savefig(HERE/(name+'.svg'),facecolor='white',bbox_inches=None)
    plt.close(fig)


def fig02(c):
    d=c['stability'].sort_values(['进入前五比例','因素排名中位数'],ascending=[False,True],kind='stable')
    chosen=set(c['top5']['物理因素']);color=[FACTOR_COLORS[v] if v in chosen else PALE for v in d['物理因素']]
    fig,(a,b)=plt.subplots(1,2,figsize=(WIDTH,4.70),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
    fig.subplots_adjust(left=.235,right=.975,bottom=.20,top=.765,wspace=.19)
    y=np.arange(len(d));a.barh(y,d['进入前五比例'],height=.53,color=color)
    for i,v in enumerate(d['进入前五比例']):
        a.text(v+.025,i,f'{100*v:.1f}'.rstrip('0').rstrip('.')+'%',va='center',fontsize=7.5,color=INK)
    labels=[v+' *' if v=='地表湍流热交换' else v for v in d['物理因素']]
    a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(0,1.25)
    a.set_xticks([0,.5,1]);a.xaxis.set_major_formatter(PercentFormatter(1))
    a.set_xlabel('前五入选率',labelpad=6)
    a.set_title('a  入选稳定性',loc='left',fontsize=8.6,pad=10,color=INK)
    for i,(_,row) in enumerate(d.iterrows()):
        b.plot([row['因素排名最优'],row['因素排名最差']],[i,i],lw=2,color=color[i],solid_capstyle='round')
    b.scatter(d['因素排名中位数'],y,s=25,c=color,edgecolor='white',lw=.5,zorder=3)
    b.set_xlim(.5,10.5);b.set_xticks([1,3,5,7,10]);b.tick_params(axis='y',left=False,labelleft=False)
    b.set_xlabel('跨场景排名（1为最优）',labelpad=6);b.set_title('b  排名中位数与范围',loc='left',fontsize=8.6,pad=10,color=INK)
    for ax in [a,b]:
        polish(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('关键因素筛选的稳健性',x=.5,y=.973,ha='center',va='top',fontsize=11,fontweight='normal')
    selected_key=tuple(Patch(facecolor=FACTOR_COLORS[v],edgecolor='none') for v in c['top5']['物理因素'])
    fig.legend(handles=[selected_key,Patch(facecolor=PALE,edgecolor='none')],
               labels=['最终前五','其他因素'],handler_map={tuple:HandlerTuple(ndivide=None,pad=.12)},
               loc='upper center',bbox_to_anchor=(.5,.899),ncol=2,frameon=False,
               fontsize=7.5,handlelength=2.2,columnspacing=2.8,borderaxespad=0)
    fig.text(.51,.068,'48个预设场景；点为排名中位数，线为最优—最差范围。',ha='center',fontsize=7,color=MUTED)
    fig.text(.51,.031,'* 湍流热交换仅12/48场景可用；入选率仍统一以48为分母。',ha='center',fontsize=7,color=MUTED)
    return fig


def fig03(c):
    top=c['top5'];grid=c['grid'];hours=[12,24,48];bands=[0,3,7,15]
    matrices=[]
    for _,meta in top.iterrows():
        p=grid[grid['物理因素']==meta['物理因素']]
        mat=p.pivot(index='评价窗口小时',columns='同期日期带天数',values='进入前五比例').reindex(index=hours,columns=bands)
        matrices.append(mat.to_numpy().ravel())
    matrix=np.array(matrices)
    fig=plt.figure(figsize=(WIDTH,3.65));ax=fig.add_axes([.235,.31,.735,.48])
    cmap=LinearSegmentedColormap.from_list('blue_paper',['#FFF9E8','#9DE9DC','#29A8C7']).with_extremes(bad='#B0B0B0')
    im=ax.pcolormesh(np.arange(13),np.arange(6),np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1,
                     edgecolors='white',linewidth=1.3,rasterized=False)
    ax.set_xlim(0,12);ax.set_ylim(5,0)
    ax.set_yticks(np.arange(5)+.5,top['物理因素'],fontsize=8)
    ax.set_xticks(np.arange(12)+.5,(['同日','±3天','±7天','±15天']*3),fontsize=6.8)
    ax.tick_params(length=0,pad=7)
    for sp in ax.spines.values():sp.set_visible(False)
    for i in range(5):
        for j in range(12):
            v=matrix[i,j]
            ax.text(j+.5,i+.5,'N/A' if np.isnan(v) else f'{v:.0%}',ha='center',va='center',fontsize=6.9,
                    color=INK)
    for x in [4,8]:ax.axvline(x,color='white',lw=5)
    for x,label in zip([2,6,10],['12小时评价窗口','24小时评价窗口','48小时评价窗口']):
        ax.text(x,-.4,label,ha='center',va='center',fontsize=8,color=INK,clip_on=False)
    header(fig,'关键因素对窗口设定的敏感性')
    fig.text(.60,.213,'历史同期日期带',ha='center',fontsize=7.5,color=MUTED)
    cax=fig.add_axes([.42,.155,.36,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cb.solids.set_rasterized(False);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=7)
    cb.ax.xaxis.set_major_formatter(PercentFormatter(1));cb.set_label('前五入选率',fontsize=7,labelpad=3)
    fig.text(.51,.018,'每格为4种QC与FLUX处理组合的汇总；共48场景。',ha='center',fontsize=7,color=MUTED)
    return fig


def fig04(c):
    fig=plt.figure(figsize=(WIDTH,5.65));gs=fig.add_gridspec(2,6,left=.105,right=.97,bottom=.145,top=.79,wspace=1.65,hspace=.9)
    positions=[(0,0,2),(0,2,4),(0,4,6),(1,1,3),(1,3,5)]
    titles=['剖面土壤含水量\n24小时变化','24小时平均\n2米气温','24小时累计\n正净辐射','72小时\n累计降水','24小时平均\n2米相对湿度']
    rng=np.random.default_rng(42)
    for i,((_,meta),(r,a,b)) in enumerate(zip(c['top5'].iterrows(),positions)):
        ax=fig.add_subplot(gs[r,a:b]);part=c['distribution'][c['distribution']['物理因素']==meta['物理因素']]
        hist=part.loc[part['样本组']=='2021—2024历史同期','数值'].dropna().to_numpy()
        event=part.loc[part['样本组']=='2026事件窗口','数值'].dropna().to_numpy()
        bp=ax.boxplot([hist,event],positions=[1,2],widths=.46,showfliers=False,patch_artist=True,
                      medianprops={'color':INK,'lw':1.1},whiskerprops={'color':'#505050','lw':.7},
                      capprops={'color':'#505050','lw':.7},boxprops={'lw':.8,'edgecolor':'#404040'})
        for patch,col in zip(bp['boxes'],['#FFE0D6','#CFEFFC']):patch.set_facecolor(col)
        for x,v,col in [(1,hist,HISTORY_COLOR),(2,event,EVENT_COLOR)]:
            ax.scatter(x+rng.normal(0,.035,len(v)),v,s=5,alpha=.5,c=col,lw=0,zorder=3)
        ax.set_xticks([1,2],[f'历史\nn={len(hist)}',f'事件\nn={len(event)}'],fontsize=7)
        ax.set_xlim(.55,2.45);ax.set_ylabel(meta['单位'],fontsize=7,labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if i==0:ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3f}'))
        ax.set_title(f'{chr(97+i)}  {titles[i]}',loc='left',fontsize=8.3,fontweight='normal',pad=10)
        polish(ax,'y')
    header(fig,'灾前关键特征与历史同期的分布差异')
    fig.legend(handles=[Patch(facecolor='#FFE0D6',edgecolor=HISTORY_COLOR,label='2021—2024年历史同期'),
                        Patch(facecolor='#CFEFFC',edgecolor=EVENT_COLOR,label='2026年事件窗口')],
               loc='upper center',bbox_to_anchor=(.51,.922),ncol=2,frameon=False,fontsize=7.4,columnspacing=2)
    fig.text(.51,.063,'箱体：Q1—Q3；横线：中位数；须线：1.5×IQR范围内最远观测；散点：全部有效值。',
             ha='center',fontsize=6.8,color=MUTED)
    fig.text(.51,.030,'严格QC口径；n为小时观测数，包含滚动特征的时间依赖，不代表独立灾害事件数。',
             ha='center',fontsize=6.8,color=MUTED)
    return fig


BUILDERS={'fig02_stability':fig02,'fig03_sensitivity':fig03,'fig04_distributions':fig04}
if __name__=='__main__':
    with mpl.rc_context(STYLE):
        data=load()
        for name,builder in BUILDERS.items():save(builder(data),name);print(name)
    provenance={'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in CACHE.iterdir() if f.is_file()},
                'design':'参考实际期刊图：明亮五因素分类色，热图浅黄至青色顺序色标，历史珊瑚红与事件天蓝；布局及数值不变',
                'exports':'SVG文字转轮廓、无栅格数据；PNG 600DPI'}
    (HERE/'来源记录.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

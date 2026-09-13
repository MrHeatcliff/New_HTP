"""Describe long training episodes and completed fixed-policy evaluation."""
import json
import os
import csv
import hashlib
from pathlib import Path


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Submit CPU analysis through Slurm'
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path('production_runs/reborn_harmony_2ew6wt0w')
    out = Path('paper_artifacts/upndown_long_episode_report')
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    def read(path):
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return [json.loads(line) for line in path.read_text().splitlines()]
    train = read(root/'harmony/up_n_down/full/paper_artifacts/episode_scores.jsonl')
    timeline = []
    previous = 0
    for row in train:
        end = int(row['agent_actions'])
        length = int(row['episode_length'])-1
        assert end-previous == length
        timeline.append(dict(episode=row['episode_index'], start=previous, end=end,
                             actions=length, score=row['episode_score']))
        previous = end
    allrows, summaries, curves = [], {}, {}
    for arm in ('full', 'harmony'):
        curves[arm] = []
        for step in range(10000,100001,10000):
            rows = read(root/arm/'evaluation/up_n_down'/f'{step:06d}'/'paper_artifacts/eval_metrics.jsonl')
            assert len(rows) == (100 if step==100000 else 10)
            score = np.array([r['episode_score'] for r in rows], dtype=float)
            length = np.array([r['episode_length']-1 for r in rows], dtype=float)
            assert np.isfinite(score).all() and np.all((length>0)&(length<=27000))
            reference = json.loads((root/arm/'evaluation/up_n_down/summary.json').read_text())[step//10000-1]
            np.testing.assert_allclose(score,reference['returns'])
            density = score.sum()/length.sum()
            curves[arm].append(dict(step=step, episodes=len(rows), mean_return=float(score.mean()),
                                    mean_actions=float(length.mean()), reward_per_action=float(density)))
            for i,(s,l) in enumerate(zip(score,length)):
                allrows.append(dict(arm=arm,checkpoint=step,episode=i+1,score=float(s),actions=int(l)))
            if step==100000:
                summaries[arm] = dict(curves[arm][-1], median_return=float(np.median(score)),
                    return_p10=float(np.quantile(score,.1)),return_p90=float(np.quantile(score,.9)),
                    median_actions=float(np.median(length)),cap_length_count=int((length==27000).sum()))
    plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    late = [r for r in timeline if r['end']>=50000]
    fig, ax = plt.subplots(figsize=(12,6))
    for x in range(50000,100001,5000):ax.axvline(x,color='#e2e8f0',lw=.8,zorder=0)
    for y,r in enumerate(late):
        ax.plot([r['start'],r['end']],[y,y],lw=5,color='#7C3AED',solid_capstyle='round')
        ax.plot(r['end'],y,'o',color='#7C3AED')
        ax.text(r['end']+500,y,f"{r['score']:,.0f} return",va='center',fontsize=8)
    ax.set_yticks(range(len(late)),[f"Episode {r['episode']} · {r['actions']:,} actions" for r in late])
    ax.set_xlim(48000,112000);ax.invert_yaxis()
    ax.set_xlabel('Cumulative training agent actions')
    ax.set_title('Up N Down · Long episodes span multiple 5K bins\nBars show duration; dots mark when completed return is logged')
    ax.axvline(100000,color='#475569',ls='--',lw=1)
    ax.text(100500,.5,'100K training budget',rotation=90,fontsize=9)
    fig.tight_layout();save(fig,'training_episode_timeline')
    fig, axes = plt.subplots(2,3,figsize=(15,8.5))
    for arm,color,label in [('full','#64748b','Reborn Full'),('harmony','#7C3AED','Reborn + Harmony')]:
        for ax,key,title in zip(axes[0],('mean_return','mean_actions','reward_per_action'),
                               ('Mean evaluation return','Mean episode actions','Total reward / total actions')):
            ax.plot([r['step'] for r in curves[arm]],[r[key] for r in curves[arm]],'o-',color=color,label=label,markersize=4)
            ax.set_title(title);ax.set_xlabel('Training agent actions');ax.grid(alpha=.2)
        final = [r for r in allrows if r['arm']==arm and r['checkpoint']==100000]
        scores=np.array([r['score'] for r in final]);lengths=np.array([r['actions'] for r in final])
        for ax,values,title in ((axes[1,0],scores,'Final return distribution'),(axes[1,1],lengths,'Final episode-length distribution')):
            ax.step(np.sort(values),np.arange(1,101)/100,where='post',color=color,label=label)
            ax.set_title(title);ax.set_ylabel('Empirical cumulative fraction');ax.grid(alpha=.2)
        axes[1,2].scatter(lengths,scores,s=16,alpha=.5,color=color,label=label)
    axes[1,0].set_xlabel('Episode return');axes[1,1].set_xlabel('Episode actions')
    axes[1,1].axvline(27000,color='#94a3b8',ls='--')
    axes[1,2].set(xlabel='Episode actions',ylabel='Episode return',title='Final evaluation: length vs return')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Up N Down · Fixed-policy evaluation, training seed 0\n10 episodes/checkpoint at 10K–90K; 100 episodes at 100K',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'evaluation_dashboard')
    full,harmony=summaries['full'],summaries['harmony']
    ratios={k:harmony[k]/full[k] for k in ('mean_return','mean_actions','reward_per_action')}
    assert np.isclose(ratios['mean_return'],ratios['mean_actions']*ratios['reward_per_action'])
    payload=dict(summary=summaries,ratios=ratios,curves=curves,timeline=timeline,sources=sources,
        length_definition='episode_length minus one reset callback',
        cap_caveat='27000-action length is a cap-hit proxy, not independently logged termination cause',
        training_seed=0,job=os.environ['SLURM_JOB_ID'])
    (out/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'evaluation_episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    text='# Up N Down — reporting long episodes and control performance\n\n'
    text+='## 1. Vì sao training curve có khoảng trống?\n\n![Training timeline](training_episode_timeline.png)\n\n'
    text+=('Mỗi return chỉ được log khi episode kết thúc; độ dài thực = episode_length − 1 callback reset. '
           'Kiểm tra tất cả episode cho thấy độ dài khớp chênh lệch action giữa hai lần kết thúc. '
           'Các bin65K–70K và75K–95K không có episode kết thúc, không phải score bằng0 hay mất training.\n\n'
           'Episode82 chạy từ64,548 đến72,274 actions, return60,540. Episode83 chạy từ72,274 đến97,236 '
           'actions, dài24,962 actions và return242,470. Timeline thể hiện toàn bộ khoảng tích lũy thay vì '
           'gán return vào từng bin đi qua. Policy vẫn cập nhật trong episode training, nên score này không '
           'phải performance của riêng checkpoint97K. Sau97,236, budget100K còn2,764 actions; '
           'không coi đoạn chưa có episode hoàn tất tiếp theo là một episode return đầy đủ.\n\n')
    text+='## 2. Evaluation bằng policy cố định\n\n![Evaluation dashboard](evaluation_dashboard.png)\n\n'
    text+=('So sánh **Reborn Full và Reborn + Harmony của cùng screening**, không phải DreamerV3. '
           'Final dùng100episode mỗi arm tại100K; checkpoint trung gian10episode. '
           'Không gọi số episode này là100training seeds.\n\n'
           '| Metric final | Full | Harmony |\n|---|---:|---:|\n')
    for name,key in [('Mean return','mean_return'),('Median return','median_return'),('Return P10','return_p10'),
                     ('Return P90','return_p90'),('Mean episode actions','mean_actions'),('Median episode actions','median_actions'),
                     ('Total reward / total actions','reward_per_action'),('Episodes dài đúng27,000 actions /100','cap_length_count')]:
        text+=f'| {name} | {full[key]:,.3f} | {harmony[key]:,.3f} |\n'
    text+=('\n27,000actions tương ứng giới hạn108,000frames với repeat4 của wrapper. '
           'Tỷ lệ chạm độ dài này là **proxy cap-hit**, không phân biệt chắc chắn timeout với game-over '
           'xảy ra cùng bước. Không có death-cause riêng trong các episode rows. Các episode chạm cap '
           'không cho biết agent sẽ sống thêm bao lâu nếu bỏ cap.\n\n'
           '## 3. Score tăng do sống lâu hay kiếm reward nhanh hơn?\n\n'
           '$$\n\overline R=\overline T\;\frac{\sum_i R_i}{\sum_i T_i}.\n$$\n\n'
           f'Tỷ lệ Harmony/Full: **return {ratios["mean_return"]:.3f}× = độ dài {ratios["mean_actions"]:.3f}× × reward/action {ratios["reward_per_action"]:.3f}×**.\n\n'
           'Reward/action ở đây là tổng reward chia tổng actions, không phải trung bình của từng tỷ số episode. '
           'Đây là đẳng thức phân rã mô tả, không chứng minh tăng độ dài gây tăng score hay component nào gây improvement. '
           'Reward/action có thể bị ảnh hưởng bởi phase game và reward schedule.\n\n'
           '## 4. Cách dùng trong báo cáo\n\n'
           '- Training curve giữ bin trống và marker tại dữ liệu thực; timeline giải thích episode dài.\n'
           '- Fixed-policy evaluation là bằng chứng chính cho control tại checkpoint; ECDF cho thấy phân phối100episode.\n'
           '- Báo cả return, length và reward/action; không chỉ chọn episode training cao nhất.\n'
           '- Kết quả chỉ training seed0, chưa xác nhận độ ổn định nhiều seed hoặc nguyên nhân riêng của Harmony.\n\n'
           '[Dữ liệu từng episode evaluation](evaluation_episodes.csv) · [Thống kê và source hashes](analysis.json)\n\n'
           '[Training curves26game](../harmony_vs_dreamerv3_26_training_curves/README.md) · '
           '[Evaluation curves26game](../harmony_vs_dreamerv3_26_learning_curves/README.md)\n\n'
           'Tái tạo: scripts/slurm_upndown_long_episode_report.sh. Chỉ phân tích CPU từ log có sẵn, không train/eval lại.\n')
    (out/'README.md').write_text(text)
    print(json.dumps(dict(status='COMPLETE',summary=summaries,ratios=ratios)))


if __name__=='__main__':main()

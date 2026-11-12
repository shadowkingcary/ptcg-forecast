# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""History descriptors, information boundaries, and subgroup weighting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.history_depth_analysis import depth_bin, history_descriptors, month_draws, paired_interval, tournament_means


def row(mid,month,a='a',b='b',x='x',z='z',y=1):
    return dict(match_id=mid,tournament_id='t'+mid,month=month,player_A=a,player_B=b,archetype_A=x,archetype_B=z,y=y)


def describe(history,hold=()):
    return {r['match_id']:r for r in history_descriptors(history,hold)}


def test_history_depth_boundaries():
    assert [depth_bin(n) for n in [0,1,9,10,49,50,900]]==['0','1-9','1-9','10-49','10-49','50+','50+']
    try:depth_bin(-1)
    except ValueError:pass
    else:raise AssertionError('Negative history accepted')


def test_monthly_counts_exclude_entire_current_month_and_later_matches():
    warm=[row('warm1','2024-01'),row('warm2','2024-06',b='c')]
    july=[row('j1','2024-07'),row('j2','2024-07')]
    august=row('aug','2024-08')
    actual=describe(warm+july+[august])
    for key in ['j1','j2']:
        assert actual[key]['prior_A']==2 and actual[key]['prior_B']==1
        assert actual[key]['prior_A_archetype']==2 and actual[key]['prior_pair']==2
    assert actual['aug']['prior_A']==4 and actual['aug']['prior_B']==3
    before=describe(warm+july)
    assert all(before[k]==actual[k] for k in before)
    more=describe(warm+july+[row('extra','2024-07')])
    assert all(before[k]==more[k] for k in before)
    reordered=describe(warm+list(reversed(july))+[august])
    assert actual==reordered


def test_october_counts_are_frozen_and_do_not_depend_on_outcomes():
    history=[row('s','2026-09')]
    hold=[row('o1','2026-10'),row('o2','2026-10'),row('new','2026-10',a='new')]
    a=describe(history,hold)
    b=describe([{**r,'y':1-r['y']} for r in history],[{**r,'y':1-r['y']} for r in hold])
    assert a==b
    assert a['o1']['prior_A']==a['o2']['prior_A']==1
    assert a['o1']['prior_pair']==a['o2']['prior_pair']==1
    assert a['new']['prior_A']==0 and a['new']['new_players']==1
    for mid in ['o1','o2','new']:assert a[mid]['history_before']=='2026-10-01'
    # Outcomes are not required by the descriptor builder at all.
    strip=lambda rows:[{k:v for k,v in r.items() if k!='y'} for r in rows]
    assert a==describe(strip(history),strip(hold))
    try:describe(history+hold)
    except ValueError:pass
    else:raise AssertionError('October entered historical updates')


def test_groups_and_pair_support_are_invariant_to_participant_order():
    history=[row('h','2024-06',a='a',b='b',x='x',z='z')]
    a=describe(history,[row('o','2026-10',a='a',b='c',x='z',z='x')])['o']
    b=describe(history,[row('o','2026-10',a='c',b='a',x='x',z='z')])['o']
    assert a['prior_A']==b['prior_B']==1
    assert a['prior_B']==b['prior_A']==0
    assert a['depth_group']==b['depth_group']=='0'
    assert a['new_players']==b['new_players']==1
    assert a['prior_pair']==b['prior_pair']==1
    assert a['prior_A_archetype']==b['prior_B_archetype']==0
    mirror=describe([row('h','2024-06',x='x',z='x')],[row('o','2026-10',x='x',z='x')])['o']
    assert mirror['prior_pair']==1  # One earlier match, not two appearances.


def test_subgroup_means_weight_tournaments_equally():
    loss=np.array([[1.,.9],[1.,.9],[1.,.9],[1.,.1]])
    ix,counts,means=tournament_means(loss,np.array([0,0,0,1]),3)
    assert ix.tolist()==[0,1] and counts.tolist()==[3,1]
    diff=means[:,0]-means[:,1]
    assert abs(diff.mean()-.5)<1e-12
    assert abs((loss[:,0]-loss[:,1]).mean()-.3)<1e-12
    result=paired_interval(diff,np.array([0,1]),2,np.tile([1,1],(100,1)))
    assert abs(result['gain']-.5)<1e-12 and np.allclose(result['ci'],[.5,.5])


def test_bootstrap_keeps_calendar_gaps_and_reports_empty_draws():
    draws=month_draws(27,replicates=8)
    assert draws.shape==(8,27) and np.all(draws.sum(axis=1)==28)
    assert np.array_equal(draws,month_draws(27,replicates=8))
    # Two supported months at indices 0 and 2; the final draw has no group data.
    sparse=np.array([[1,0,1,0,0],[0,0,0,0,1]])
    result=paired_interval([.2,.6],[0,2],5,sparse)
    assert result['zero_support_replicates']==1 and result['ci'] is None
    assert abs(result['gain']-.4)<1e-12


def test_one_unseen_and_two_unseen_groups_are_distinct():
    hist=[row('h','2024-06')]
    d=describe(hist,[row('one','2026-10',a='new1'),row('both','2026-10',a='new1',b='new2')])
    assert d['one']['depth_group']==d['both']['depth_group']=='0'
    assert d['one']['new_players']==1 and d['both']['new_players']==2


def test_october_endpoints_preserve_existing_sign_reversal_convention():
    draws=np.array([[i,100-i] for i in range(101)])
    positive=paired_interval([0.,1.],[0,1],2,draws,reverse_endpoints=True)
    opposite=paired_interval([0.,-1.],[0,1],2,draws)
    assert positive['ci']==[-opposite['ci'][1],-opposite['ci'][0]]
    assert positive['gain']==-opposite['gain']

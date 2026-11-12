# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Join raw matches, apply eligibility rules, and pseudonymize both periods together.
Writes candidate unified tables and audits outside the repo; frozen paper inputs are read-only.
"""
import argparse,csv,gzip,io,json,os,secrets
from collections import Counter,defaultdict
from pathlib import Path
from project_paths import ROOT,WORK,REFERENCE,external
from pipeline_utils import (FIELDS,ROTATION_BOUNDARIES,SPLIT,date_range,read_endpoint,
                           save_json,sha256,utcnow,validate_tournament,validate_array)


def extract_rows(tournament,pairings,standings):
    validate_tournament(tournament)
    tid,date=tournament['id'],tournament['date']
    pairings=validate_array(pairings,'pairings');standings=validate_array(standings,'standings')
    archetypes={};fallback=0
    for entry in standings:
        player=entry.get('player');deck=entry.get('deck') or {}
        if not isinstance(deck,dict): raise ValueError('Malformed deck metadata')
        archetype=deck.get('id') or deck.get('name')
        if player:
            if not isinstance(player,str): raise ValueError('Player ID must be a string')
            if player in archetypes: raise ValueError('Duplicate player in standings')
            archetypes[player]=archetype
            fallback+=bool(archetype and not deck.get('id'))
    rows=[];excluded=Counter();seen={};self_pairings=0
    for m in pairings:
        a,b,w=m.get('player1'),m.get('player2'),m.get('winner')
        if not b: excluded['bye_or_missing_second_player']+=1;continue
        if not a: excluded['missing_first_player']+=1;continue
        if w==0: excluded['tie']+=1;continue
        if w not in (a,b): excluded['double_loss' if w==-1 else 'unresolved']+=1;continue
        if not archetypes.get(a) or not archetypes.get(b): excluded['missing_archetype']+=1;continue
        if m.get('round') is None: raise ValueError('Invalid decisive match')
        if not isinstance(a,str) or not isinstance(b,str): raise ValueError('Player ID must be a string')
        round_no=str(m['round'])
        if not round_no.isdigit(): raise ValueError('Invalid round number')
        mid=f'{tid}_{round_no}_{a}_{b}'
        if m.get('match'): mid+=f"_{m['match']}"
        row=dict(zip(FIELDS,[tid,date,str(1+sum(date[:10]>=x for x in ROTATION_BOUNDARIES)),mid,
                            a,b,archetypes[a],archetypes[b],'A' if w==a else 'B',round_no]))
        if mid in seen:
            if seen[mid]!=row: raise ValueError('Conflicting duplicate match')
            excluded['duplicate_exact']+=1;continue
        seen[mid]=row;rows.append(row);self_pairings+=int(a==b)
    assert len(rows)+sum(excluded.values())==len(pairings)
    return rows,{'pairings':len(pairings),'eligible_matches':len(rows),'standings_rows':len(standings),
                 'deck_name_fallback_entries':fallback,'self_pairings_retained':self_pairings,'excluded':dict(excluded)}


def assign_tournaments(mapping, dates, metadata):
    """Keep old surrogates and the original ordering within identical timestamps."""
    used = set(mapping.values())
    for item in metadata:
        if item["id"] in dates and dates[item["id"]] != item["date"]:
            raise ValueError("API tournament date changed from the original table")
        dates[item["id"]] = item["date"]
    groups = defaultdict(list)
    for tid, date in dates.items():
        groups[date].append(tid)
    for tids in groups.values():
        ordered = sorted(tids)
        anchors = [(-1, -1)] + [(i, int(mapping[t][1:], 16)) for i, t in enumerate(ordered) if t in mapping]
        anchors.append((len(ordered), 16 ** 12))
        for (left_i, low), (right_i, high) in zip(anchors, anchors[1:]):
            if high <= low:
                raise ValueError("Existing tied-timestamp tournament order changed")
            gap = ordered[left_i + 1:right_i]
            if high - low - 1 < len(gap):
                raise ValueError("No surrogate space between existing tournament anchors")
            candidates = set()
            attempts = 0
            while len(candidates) < len(gap):
                attempts += 1
                if attempts > 100000:
                    raise ValueError("Unable to allocate collision-free tournament surrogates")
                value = "t" + f"{low + 1 + secrets.randbelow(high - low - 1):012x}"
                if value not in used:
                    candidates.add(value)
            for tid, value in zip(gap, sorted(candidates)):
                mapping[tid] = value
                used.add(value)


def remap(row,mapping):
    out=dict(row);a=mapping['players'][row['player_A']];b=mapping['players'][row['player_B']]
    t=mapping['tournaments'][row['tournament_id']]
    prefix=f"{row['tournament_id']}_{row['round']}_{row['player_A']}_{row['player_B']}"
    if not row['match_id'].startswith(prefix): raise ValueError('Unexpected match ID layout')
    out.update(tournament_id=t,player_A=a,player_B=b,match_id=f"{t}_{row['round']}_{a}_{b}"+row['match_id'][len(prefix):])
    return out


def write_gzip(path,rows):
    buffer=io.StringIO(newline='');w=csv.DictWriter(buffer,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    content=buffer.getvalue().encode('utf-8')
    if path.exists():
        if gzip.decompress(path.read_bytes())!=content: raise FileExistsError('Use a new output directory; refusing to replace different data: '+str(path))
        return
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(gzip.compress(content,mtime=0));temp.chmod(0o600);temp.replace(path)


def build(cache,out,map_path,split=SPLIT,preserve_labels_from=()):
    cache=external(cache);out=external(out);map_path=external(map_path)
    meta=json.loads((cache/'tournaments.json').read_text());acq=json.loads((cache/'acquisition.json').read_text())
    if sha256(cache/'tournaments.json')!=acq['listing_sha256']: raise ValueError('Listing digest mismatch')
    date_range(acq['start_inclusive'],split);date_range(split,acq['end_exclusive'])
    seen_t={}
    for t in meta:
        validate_tournament(t)
        if not acq['start_inclusive']<=t['date'][:10]<acq['end_exclusive']: raise ValueError('Event outside requested range')
        if t['id'] in seen_t and seen_t[t['id']]!=t: raise ValueError('Conflicting duplicate event')
        seen_t[t['id']]=t
    meta=sorted(seen_t.values(),key=lambda t:(t['date'],t['id']))
    if preserve_labels_from and not map_path.exists():
        raise ValueError('Preserving de-identified labels requires the corresponding existing identifier map')
    frozen_labels={};label_sources={}
    for source in preserve_labels_from:
        source=Path(source);label_sources[str(source.resolve())]=sha256(source)
        opener=gzip.open if source.suffix=='.gz' else open
        with opener(source,'rt',newline='') as stream:
            for r in csv.DictReader(stream):
                for side in ('A','B'):
                    key=(r['tournament_id'],r['player_'+side]);label=r['archetype_'+side]
                    if frozen_labels.setdefault(key,label)!=label:
                        raise ValueError('Conflicting frozen labels for a player/event')
    label_changes=set()
    mapping=json.loads(map_path.read_text()) if map_path.exists() else {'version':1,'created_utc':utcnow(),'players':{},'tournaments':{},'tournament_dates':{}}
    for field,prefix in [('players','p'),('tournaments','t')]:
        vals=list(mapping[field].values())
        if len(vals)!=len(set(vals)): raise ValueError('Identifier mapping is not one-to-one')
        if any(not isinstance(v,str) or len(v)!=13 or v[0]!=prefix or any(c not in '0123456789abcdef' for c in v[1:]) for v in vals):
            raise ValueError('Invalid surrogate identifier')
    assign_tournaments(mapping['tournaments'],mapping['tournament_dates'],meta)
    save_json(map_path,mapping)
    used=set(mapping['players'].values());rows=[];audits=[];excluded=Counter();seen=set()
    for i,t in enumerate(meta,1):
        event,audit=extract_rows(t,read_endpoint(cache,t['id'],'pairings'),read_endpoint(cache,t['id'],'standings'))
        for row in event:
            for key in ('player_A','player_B'):
                player=row[key]
                if player not in mapping['players']:
                    value='p'+secrets.token_hex(6)
                    while value in used: value='p'+secrets.token_hex(6)
                    mapping['players'][player]=value;used.add(value)
            r=remap(row,mapping)
            for side in ('A','B'):
                key=(r['tournament_id'],r['player_'+side]);field='archetype_'+side
                if key in frozen_labels and r[field]!=frozen_labels[key]:
                    label_changes.add(key);r[field]=frozen_labels[key]
            if r['match_id'] in seen: raise ValueError('Duplicate final match ID')
            seen.add(r['match_id']);rows.append(r)
        excluded.update(audit['excluded']);audits.append({'tournament_id':mapping['tournaments'][t['id']],'date':t['date'],**audit})
        if i%250==0: save_json(map_path,mapping);print(f'Joined {i}/{len(meta)} events',flush=True)
    save_json(map_path,mapping)
    rows.sort(key=lambda r:(r['tournament_date'],r['tournament_id'],int(r['round']),r['match_id']))
    history=[r for r in rows if r['tournament_date'][:10]<split]
    holdout=[r for r in rows if r['tournament_date'][:10]>=split]
    if not history or not holdout: raise ValueError('Both history and holdout need eligible matches')
    if {r['tournament_id'] for r in history}&{r['tournament_id'] for r in holdout}: raise ValueError('Event crosses period boundary')
    out.mkdir(parents=True,exist_ok=True)
    protected={p:sha256(p) for p in (ROOT/'data/unified').glob('*.gz')}
    outputs={'matches.csv.gz':history,'matches_oct2026_holdout.csv.gz':holdout}
    for name,table in outputs.items(): write_gzip(out/name,table)
    def stats(table):
        return {'matches':len(table),'tournaments':len({r['tournament_id'] for r in table}),
                'players':len({r[s] for r in table for s in ['player_A','player_B']}),
                'archetypes':len({r[s] for r in table for s in ['archetype_A','archetype_B']}),
                'min_date':min(r['tournament_date'] for r in table),'max_date':max(r['tournament_date'] for r in table)}
    report={'completed_utc':utcnow(),'split_date':split,'listed_tournaments':len(meta),'history':stats(history),'holdout':stats(holdout),
            'exclusions':dict(excluded),'self_pairings_retained':sum(a['self_pairings_retained'] for a in audits),'tournaments':audits,'source_acquisition_sha256':sha256(cache/'acquisition.json'),
            'identifier_map_sha256':sha256(map_path),'preserved_label_sources':label_sources,
            'player_event_labels_preserved':len(label_changes),'outputs':{name:sha256(out/name) for name in outputs},
            'checks':{'unique_match_ids':True,'one_mapping_across_periods':True,'no_tournament_overlap':True,'all_pairings_accounted_for':True},
            'snapshot_policy':'Use archived endpoint contents. Later API responses can change eligibility and deck labels; candidates do not overwrite frozen paper inputs.'}
    expected=json.loads(REFERENCE.read_text())['dataset']
    report['paper_population_comparison']={label:{'expected_matches':expected[key]['matches'],'actual_matches':len(table),'matches_equal':expected[key]['matches']==len(table)}
        for label,key,table in [('history','history',history),('holdout','october',holdout)]}
    assert protected=={p:sha256(p) for p in protected},'Frozen input changed'
    save_json(out.parent/'build_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='tournaments'},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=WORK/'raw')
    p.add_argument('--output-dir',type=Path,default=WORK/'rebuilt/unified')
    p.add_argument('--identifier-map',type=Path,default=WORK/'private/identifier_map.json')
    p.add_argument('--split-date',default=SPLIT)
    p.add_argument('--preserve-labels-from',type=Path,action='append',default=[])
    a=p.parse_args();os.umask(0o077);build(a.cache_dir,a.output_dir,a.identifier_map,a.split_date,a.preserve_labels_from)

if __name__=='__main__':main()

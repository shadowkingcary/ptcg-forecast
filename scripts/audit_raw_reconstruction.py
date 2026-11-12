# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Compare private archived raw snapshots with the two frozen unified datasets.
The corresponding private identifier map is required to compare exact rows.
Only counts, hashes, and exclusion summaries are written to the public-facing audit.
"""
import argparse,csv,gzip,hashlib,json,os
from pathlib import Path
from project_paths import ROOT,WORK,external
from pipeline_utils import FIELDS,START,END,SPLIT,sha256,save_json
from build_unified import extract_rows,remap


def row_digest(row):
    return hashlib.sha256(json.dumps([str(row[k]) for k in FIELDS],ensure_ascii=False).encode()).digest()


def audit_period(listing,cache,reference,mapping,start,end):
    expected={};labels={}
    with gzip.open(reference,'rt',newline='') as stream:
        for row in csv.DictReader(stream):
            if not start<=row['tournament_date'][:10]<end: continue
            if row['match_id'] in expected: raise ValueError('Duplicate frozen match')
            expected[row['match_id']]=row_digest(row)
            for side in ('A','B'):
                key=(row['tournament_id'],row['player_'+side]);value=row['archetype_'+side]
                if labels.setdefault(key,value)!=value: raise ValueError('Inconsistent frozen label')
    metadata={}
    for t in json.loads(listing.read_text()):
        if start<=t['date'][:10]<end:
            if t['id'] in metadata and metadata[t['id']]!=t: raise ValueError('Conflicting listing metadata')
            metadata[t['id']]=t
    n=changed=missing_cache=self_pairings=0;seen=set();preserved=set();excluded={};endpoint_hashes={}
    for t in metadata.values():
        pairings=cache/'pairings'/(t['id']+'.json');standings=cache/'standings'/(t['id']+'.json')
        if not pairings.exists() or not standings.exists(): missing_cache+=1;continue
        # Hash legacy snapshots even when their original collection predates sidecar provenance.
        for path in (pairings,standings): endpoint_hashes[str(path.relative_to(cache))]=sha256(path)
        rows,details=extract_rows(t,json.loads(pairings.read_text()),json.loads(standings.read_text()))
        self_pairings+=details['self_pairings_retained']
        for key,value in details['excluded'].items(): excluded[key]=excluded.get(key,0)+value
        for row in rows:
            r=remap(row,mapping);n+=1
            for side in ('A','B'):
                key=(r['tournament_id'],r['player_'+side]);field='archetype_'+side
                if key in labels and r[field]!=labels[key]: preserved.add(key);r[field]=labels[key]
            if r['match_id'] in seen: raise ValueError('Duplicate rebuilt match')
            seen.add(r['match_id']);changed+=row_digest(r)!=expected.get(r['match_id'])
    missing=len(set(expected)-seen)
    return {'eligible_matches':n,'expected_matches':len(expected),'missing_cached_events':missing_cache,
            'missing_published_matches':missing,'changed_or_extra_rows_after_label_preservation':changed,
            'preserved_player_event_labels':len(preserved),'self_pairings_retained':self_pairings,
            'exclusions':excluded,'listing_sha256':sha256(listing),'reference_sha256':sha256(reference),
            'endpoint_hash_manifest_sha256':hashlib.sha256(json.dumps(endpoint_hashes,sort_keys=True).encode()).hexdigest(),
            'passed':missing_cache==0 and missing==0 and changed==0 and n==len(expected)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history-listing',type=Path,default=WORK/'raw/tournaments.json')
    p.add_argument('--history-cache',type=Path,default=WORK/'raw')
    p.add_argument('--october-listing',type=Path,default=WORK/'raw/tournaments.json')
    p.add_argument('--october-cache',type=Path,default=WORK/'raw')
    p.add_argument('--identifier-map',type=Path,default=WORK/'private/identifier_map.json')
    p.add_argument('--output',type=Path,default=WORK/'raw_reconstruction_audit.json')
    a=p.parse_args();os.umask(0o077);out=external(a.output)
    mapping=json.loads(external(a.identifier_map).read_text())
    reports={}
    for label,listing,cache,reference,start,end in [
        ('history',a.history_listing,a.history_cache,ROOT/'data/unified/matches.csv.gz',START,SPLIT),
        ('october',a.october_listing,a.october_cache,ROOT/'data/unified/matches_oct2026_holdout.csv.gz',SPLIT,END)]:
        reports[label]=audit_period(listing,external(cache),reference,mapping,start,end)
        print(label,json.dumps(reports[label]),flush=True)
    save_json(out,reports)
    if not all(r['passed'] for r in reports.values()): raise SystemExit('Raw reconstruction differs; inspect '+str(out))

if __name__=='__main__':main()

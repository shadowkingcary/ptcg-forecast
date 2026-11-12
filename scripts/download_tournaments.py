# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Download a complete date-bounded Standard tournament cache outside the repo.
Supports resumable checked caches, listing snapshots, and explicit local imports.
"""
import argparse,json,os
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from project_paths import WORK,external
from pipeline_utils import (BASE,START,END,Client,date_range,read_endpoint,
    save_json,sha256,utcnow,validate_array,validate_tournament)


def list_tournaments(client,run_dir,start,end,max_pages=10000):
    found={};seen_pages=set();previous=None
    for page in range(1,max_pages+1):
        listing,provenance=client.get(f'{BASE}/tournaments?game=PTCG&format=STANDARD&limit=100&page={page}')
        if not isinstance(listing,list): raise ValueError('Tournament listing is not an array')
        for item in listing: validate_tournament(item)
        dates=[t['date'] for t in listing]
        if dates!=sorted(dates,reverse=True): raise ValueError('Listing is not newest-first')
        if dates and previous is not None and dates[0]>previous:
            raise ValueError('Listing shifted during pagination; restart the listing')
        signature=tuple(t['id'] for t in listing)
        if signature and signature in seen_pages: raise ValueError('Repeated listing page')
        seen_pages.add(signature)
        save_json(run_dir/f'page_{page}.json',listing)
        provenance['sha256']=sha256(run_dir/f'page_{page}.json')
        save_json(run_dir/f'page_{page}.provenance.json',provenance)
        for item in listing:
            if start<=item['date'][:10]<end:
                tid=item['id']
                if tid in found and found[tid]!=item: raise ValueError('Conflicting duplicate metadata')
                found[tid]=item
        print(f'Listing page {page}: {len(found)} eligible-period events',flush=True)
        if not dates or dates[-1][:10]<start:
            return sorted(found.values(),key=lambda t:(t['date'],t['id']))
        previous=dates[-1]
    raise RuntimeError('Pagination cap reached before start date; no complete cache claimed')


def download(cache,start=START,end=END,*,reuse_listing=False,listing_file=None,
             import_cache=None,offline=False,max_pages=10000,client=None):
    cache=external(cache);date_range(start,end);cache.mkdir(parents=True,exist_ok=True)
    client=client or Client();listing_path=cache/'tournaments.json'
    config={'start_inclusive':start,'end_exclusive':end,'game':'PTCG','format':'STANDARD'}
    config_path=cache/'request.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Cache belongs to another date range; use a new directory')
    save_json(config_path,config)
    if reuse_listing or listing_file:
        source=Path(listing_file) if listing_file else listing_path
        raw=json.loads(source.read_text());tournaments={}
        if not isinstance(raw,list): raise ValueError('Listing snapshot must be an array')
        for item in raw:
            validate_tournament(item)
            if not start<=item['date'][:10]<end: continue
            tid=item['id']
            if tid in tournaments and tournaments[tid]!=item: raise ValueError('Conflicting duplicate metadata')
            tournaments[tid]=item
        tournaments=sorted(tournaments.values(),key=lambda t:(t['date'],t['id']))
        listing_source={'kind':'provided_snapshot' if listing_file else 'resumed_snapshot',
                        'sha256':sha256(source)}
    else:
        if offline: raise ValueError('--offline requires --listing-file or --reuse-listing')
        run_dir=cache/'listings'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        tournaments=list_tournaments(client,run_dir,start,end,max_pages)
        listing_source={'kind':'api_pagination','directory':str(run_dir)}
    if not tournaments: raise ValueError('No tournaments in requested date range')
    save_json(listing_path,tournaments)
    counts=Counter()
    for i,t in enumerate(tournaments,1):
        for endpoint in ('pairings','standings'):
            path=cache/endpoint/(t['id']+'.json');prov=path.with_suffix('.provenance.json')
            if path.exists() and prov.exists():
                read_endpoint(cache,t['id'],endpoint);counts['resumed']+=1;continue
            original=Path(import_cache)/endpoint/path.name if import_cache else None
            if original is not None and original.exists():
                data=json.loads(original.read_text());validate_array(data,endpoint)
                provenance={'source':'imported_cache','imported_utc':utcnow(),'source_sha256':sha256(original)}
                counts['imported']+=1
            else:
                if offline: raise FileNotFoundError('Missing offline endpoint: '+str(path))
                data,provenance=client.get(f"{BASE}/tournaments/{t['id']}/{endpoint}")
                validate_array(data,endpoint);provenance['source']='api';counts['downloaded']+=1
            save_json(path,data);provenance['sha256']=sha256(path);save_json(prov,provenance)
        if i%25==0 or i==len(tournaments): print(f'Cached {i}/{len(tournaments)} events: {dict(counts)}',flush=True)
    report={**config,'completed_utc':utcnow(),'listed_tournaments':len(tournaments),
            'listing_sha256':sha256(listing_path),'listing_source':listing_source,
            'cached_endpoint_files':2*len(tournaments),'counts_this_run':dict(counts),
            'api_documentation':'https://docs.limitlesstcg.com/developer/tournaments'}
    save_json(cache/'acquisition.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=WORK/'raw')
    p.add_argument('--start',default=START);p.add_argument('--end',default=END)
    p.add_argument('--reuse-listing',action='store_true');p.add_argument('--listing-file',type=Path)
    p.add_argument('--import-cache',type=Path);p.add_argument('--offline',action='store_true')
    p.add_argument('--max-pages',type=int,default=10000)
    a=p.parse_args();os.umask(0o077)
    print(json.dumps(download(a.cache_dir,a.start,a.end,reuse_listing=a.reuse_listing,
        listing_file=a.listing_file,import_cache=a.import_cache,offline=a.offline,max_pages=a.max_pages),indent=2))

if __name__=='__main__':main()

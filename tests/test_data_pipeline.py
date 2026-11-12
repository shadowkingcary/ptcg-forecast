# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Raw-to-unified eligibility, provenance, linkage and publication-boundary tests."""
import contextlib,csv,gzip,io,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from project_paths import ROOT,external
from download_tournaments import download,list_tournaments
from build_unified import build,extract_rows,write_gzip
from pipeline_utils import save_json


def event(tid='event1',day='2026-09-30'):
    return dict(id=tid,date=day+'T12:00:00.000Z',game='PTCG',format='STANDARD',players=2)


def match(winner='player-a',**extra):
    return dict(player1='player-a',player2='player-b',winner=winner,round=1,**extra)


def standings():
    return [dict(player='player-a',deck={'id':'deck-x'}),dict(player='player-b',deck={'id':'deck-y'})]


def fails(kind,fn):
    try: fn()
    except kind: return
    raise AssertionError('Expected '+kind.__name__)


def fixture(root):
    incoming=root/'incoming';listing=root/'listing.json'
    meta=[event(),event('event2','2026-10-01'),event('event3','2026-10-31')]
    save_json(listing,meta)
    for t in meta:
        save_json(incoming/'pairings'/(t['id']+'.json'),[match()])
        save_json(incoming/'standings'/(t['id']+'.json'),standings())
    return incoming,listing


def test_raw_eligibility_and_bracket_rematches():
    data=[match(),match(),match(match='F'),match(winner=0),match(winner=-1),match(winner=None),
          dict(player1='player-a',player2=None,winner='player-a',round=1),
          dict(player1=None,player2='player-b',winner='player-b',round=1),
          dict(player1='unknown',player2='player-b',winner='player-b',round=1)]
    rows,audit=extract_rows(event(),data,standings())
    assert len(rows)==2 and rows[1]['match_id'].endswith('_F')
    assert audit['excluded']=={'duplicate_exact':1,'tie':1,'double_loss':1,'unresolved':1,
                              'bye_or_missing_second_player':1,'missing_first_player':1,'missing_archetype':1}
    assert audit['pairings']==audit['eligible_matches']+sum(audit['excluded'].values())


def test_conflicting_matches_and_player_joins_fail():
    fails(ValueError,lambda:extract_rows(event(),[match(),match(winner='player-b')],standings()))
    fails(ValueError,lambda:extract_rows(event(),[match()],standings()+standings()[:1]))


def test_name_fallback_and_era_assignment():
    s=standings();s[0]['deck']={'name':'fallback-category'}
    rows,audit=extract_rows(event(day='2024-03-20'),[match()],s)
    assert rows[0]['era']=='1' and rows[0]['archetype_A']=='fallback-category'
    assert audit['deck_name_fallback_entries']==1
    rows,_=extract_rows(event(day='2024-03-21'),[match()],s)
    assert rows[0]['era']=='2'


def test_offline_pipeline_stable_linkage_split_and_rerun():
    with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
        root=Path(td);incoming,listing=fixture(root);cache=root/'cache';out=root/'rebuilt/unified';mapping=root/'private/map.json'
        report=download(cache,listing_file=listing,import_cache=incoming,offline=True)
        assert report['counts_this_run']=={'imported':6}
        resumed=download(cache,reuse_listing=True,offline=True)
        assert resumed['counts_this_run']=={'resumed':6}
        first=build(cache,out,mapping);before={p.name:p.read_bytes() for p in out.iterdir()}
        second=build(cache,out,mapping)
        assert before=={p.name:p.read_bytes() for p in out.iterdir()}
        with gzip.open(out/'matches.csv.gz','rt') as f: history=list(csv.DictReader(f))
        with gzip.open(out/'matches_oct2026_holdout.csv.gz','rt') as f: hold=list(csv.DictReader(f))
        assert len(history)==1 and len(hold)==2
        assert all(r['player_A']==history[0]['player_A'] for r in hold)
        assert history[0]['player_A']!='player-a'
        assert all(r['tournament_date']<'2026-10-01' for r in history)
        assert all('2026-10-01'<=r['tournament_date']<'2026-11-01' for r in hold)
        assert all(first['checks'].values()) and first['outputs']==second['outputs']


def test_missing_offline_inputs_and_cache_tampering_fail():
    with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
        root=Path(td);incoming,listing=fixture(root);cache=root/'cache'
        fails(FileNotFoundError,lambda:download(cache,listing_file=listing,offline=True))
        download(cache,listing_file=listing,import_cache=incoming,offline=True)
        (cache/'pairings/event1.json').write_text('[]')
        fails(ValueError,lambda:download(cache,reuse_listing=True,offline=True))
        fails(ValueError,lambda:build(cache,root/'rebuilt/unified',root/'map.json'))


class PageClient:
    def __init__(self,pages): self.pages=iter(pages)
    def get(self,url): return next(self.pages),{'url':url,'retrieved_utc':'fixture'}


def test_pagination_deduplicates_and_covers_inclusive_start():
    pages=[[event('new','2026-11-01'),event('b','2026-10-31')],
           [event('b','2026-10-31'),event('a','2026-10-01')],
           [event('old','2026-09-30')]]
    with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
        result=list_tournaments(PageClient(pages),Path(td),'2026-10-01','2026-11-01')
    assert [r['id'] for r in result]==['a','b']


def test_pagination_detects_shift_and_incomplete_range():
    with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
        fails(ValueError,lambda:list_tournaments(PageClient([[event('a','2026-10-01')],[event('b','2026-10-02')]]),Path(td),'2026-10-01','2026-11-01'))
        fails(RuntimeError,lambda:list_tournaments(PageClient([[event('a','2026-10-01')]]),Path(td),'2026-10-01','2026-11-01',max_pages=1))


def test_output_boundary_and_non_overwrite():
    fails(ValueError,lambda:external(ROOT/'data/raw'))
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'matches.csv.gz';rows,_=extract_rows(event(),[match()],standings())
        write_gzip(path,rows);before=path.read_bytes();changed=[dict(rows[0],winner='B')]
        fails(FileExistsError,lambda:write_gzip(path,changed))
        assert path.read_bytes()==before


def test_self_pairing_is_flagged_without_changing_frozen_eligibility():
    rows,audit=extract_rows(event(),[dict(player1='player-a',player2='player-a',winner='player-a',round=6)],standings())
    assert len(rows)==1 and rows[0]['player_A']==rows[0]['player_B']
    assert audit['self_pairings_retained']==1


def test_frozen_labels_require_map_and_preserve_snapshot():
    from pipeline_utils import sha256
    with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
        root=Path(td);incoming,listing=fixture(root);cache=root/'cache';mapping=root/'private/map.json'
        download(cache,listing_file=listing,import_cache=incoming,offline=True)
        out=root/'first/unified';build(cache,out,mapping)
        changed=standings();changed[0]['deck']={'id':'reclassified-deck'}
        source=cache/'standings/event2.json';save_json(source,changed)
        prov=source.with_suffix('.provenance.json');record=json.loads(prov.read_text());record['sha256']=sha256(source);save_json(prov,record)
        fails(ValueError,lambda:build(cache,root/'bad/unified',root/'missing_map.json',preserve_labels_from=[out/'matches_oct2026_holdout.csv.gz']))
        report=build(cache,root/'second/unified',mapping,preserve_labels_from=[out/'matches_oct2026_holdout.csv.gz'])
        assert report['player_event_labels_preserved']==1
        assert (out/'matches_oct2026_holdout.csv.gz').read_bytes()==(root/'second/unified/matches_oct2026_holdout.csv.gz').read_bytes()

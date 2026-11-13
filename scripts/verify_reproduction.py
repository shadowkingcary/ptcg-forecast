# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Validate inputs and regenerated paper results against a frozen numerical reference."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import argparse, csv, gzip, hashlib, importlib.metadata, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=RESULTS
REFERENCE=REFERENCE

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def check_inputs():
    expected=json.loads(REFERENCE.read_text())
    for name,digest in expected['inputs'].items():
        p=ROOT/name if '/' in name else ROOT/'data/unified'/name
        assert p.is_file(), f'Missing input: {p}'
        assert sha(p)==digest, f'Input checksum mismatch: {name}'
        with gzip.open(p,'rt',newline='') as f:
            fields=next(csv.reader(f))
        required={'tournament_id','tournament_date','match_id','player_A','player_B','archetype_A','archetype_B','winner','era','round'}
        assert required <= set(fields), f'Missing schema fields in {name}'
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        name,version=line.split('==')
        actual=importlib.metadata.version(name)
        assert actual==version, f'{name}: expected {version}, installed {actual}'
    print('PASS: two input checksums, schemas and pinned dependencies',flush=True)
    return expected

def compare(expected,actual,path='results',errors=None):
    if errors is None: errors=[]
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k not in actual: errors.append(path+'.'+k+': missing')
            else: compare(v,actual[k],path+'.'+k,errors)
    elif isinstance(expected,list):
        if len(expected)!=len(actual): errors.append(path+': length differs')
        else:
            for i,(e,a) in enumerate(zip(expected,actual)): compare(e,a,f'{path}[{i}]',errors)
    elif isinstance(expected,float):
        if not math.isclose(expected,actual,abs_tol=1e-10,rel_tol=0): errors.append(f'{path}: {expected} != {actual}')
    elif expected!=actual: errors.append(f'{path}: values differ')
    return errors

def verify():
    expected=check_inputs()
    result=json.loads((P/'full_october_2026_run/paper_results.json').read_text())
    errors=[]
    for key in ('dataset','primary','later','october'):
        compare(expected[key],result[key],key,errors)
    tables=json.loads((P/'full_october_2026_run/manuscript_tables.json').read_text())
    assert set(tables)==set('123456'), 'Expected exactly six main-paper tables'
    compare(expected['tables'],tables,'tables',errors)
    weight=json.loads((P/'player_record_weight/summary.json').read_text())
    assert weight['selection']['weight']==expected['weight_selected']
    assert weight['october_outcomes_removed_prediction_identity']
    tuning=json.loads((P/'tuning_objective/summary.json').read_text())
    assert tuning['all_reported_forecasts_and_comparisons_unchanged']
    assert tuning['selection']['changed_models']==[] and tuning['refitted_evaluation_models']==[]
    assert tuning['selection']['settings']['match']==tuning['selection']['settings']['tournament']
    assert tuning['paper_results_sha256']==sha(P/'full_october_2026_run/paper_results.json')
    tuning_manifest=json.loads((P/'tuning_objective/manifest.json').read_text())
    for name,digest in tuning_manifest['output_sha256'].items():
        assert sha(P/'tuning_objective'/name)==digest, 'Tuning audit output changed: '+name
    history=json.loads((P/'history_depth/summary.json').read_text())
    assert history['validation']['all_sample_scores_and_intervals_match_existing_ledger']
    with (P/'FINAL_rating_benchmarks_results.csv').open() as f:
        elo=[r for r in csv.DictReader(f) if r['spec']=='primary' and r['reading']=='snapshot' and r['reference']=='Player-Skill' and r['model']=='Elo + archetype']
    assert len(elo)==1
    compare(expected['elo'],elo[0],'elo',errors)
    figures=['journal_experimental_design.png','journal_main_gains.png','journal_history_depth.png','journal_sharpness_calibration.png']
    for name in figures:
        path=FIGURES/name
        assert path.is_file() and path.stat().st_size>1000, f'Missing figure: {name}'
    report={'completed_utc':datetime.now(timezone.utc).isoformat(),'passed':not errors,
            'reference_sha256':sha(REFERENCE),'tolerance':1e-10,'tables_exact':tables==expected['tables'],
            'verified_periods':['primary','later','october'],'verified_models':5,
            'selected_weight':weight['selection']['weight'],'october_outcome_exclusion':True,
            'tuning_objectives_select_identical_settings':True,
            'figures':{n:sha(FIGURES/n) for n in figures},'errors':errors}
    path=P/'full_october_2026_run/reproduction_verification.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    if errors: raise AssertionError('Reproduction mismatch:\n'+'\n'.join(errors[:30]))
    print('PASS: dataset counts, five models in all periods, six exact tables, Elo, weight, tuning-objective and history-depth checks; four figures present')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only',action='store_true')
    args=parser.parse_args()
    check_inputs() if args.inputs_only else verify()

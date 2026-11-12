# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Reproduce current-paper analyses, six tables and four figures from frozen inputs."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=RESULTS/'full_october_2026_run'
sys.path.insert(0,str(ROOT/'scripts'))
from verify_reproduction import check_inputs, sha

def now(): return datetime.now(timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2,help='Concurrent analyses; use 1 for lower memory use')
    parser.add_argument('--check-inputs',action='store_true',help='Check data and dependencies without running analyses')
    args=parser.parse_args()
    if args.jobs<1: parser.error('--jobs must be positive')
    check_inputs()
    if args.check_inputs: return
    OUT.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    jobs={}
    def add(name,script,deps=()):
        jobs[name]={'command':[sys.executable,'-u','scripts/'+script+'.py'],'dependencies':list(deps),'status':'pending'}
    add('tests','run_tests')
    add('logistic','logistic_baseline',['tests'])
    add('pair','archetype_pair',['tests'])
    add('pooled','player_records',['tests'])
    add('overall','playerskill_general_only',['tests'])
    add('joint','joint_hierarchical_model',['logistic','pooled','overall'])
    add('elo','rating_benchmarks',['pair','pooled','overall'])
    add('calibration','posthoc_calibration',['joint'])
    add('october','holdout_new_methods',['calibration'])
    add('player_updates','rating_fresh_comparison',['pooled'])
    add('pair_updates','card_only_fresh_check',['pair'])
    add('monthly_equivalence','check_ps_fresh_equivalence',['pooled'])
    add('ledger','summarize_full_october',['logistic','pair','pooled','overall','joint','october'])
    add('history_depth','history_depth_analysis',['ledger'])
    add('weights','player_record_weight_sensitivity',['ledger'])
    add('tables','render_journal_tables',['history_depth','calibration','player_updates','pair_updates'])
    add('figures','journal_figures',['history_depth'])
    add('history_figure','history_depth_figures',['history_depth'])
    add('verification','verify_reproduction',['tables','figures','history_figure','weights','elo','monthly_equivalence'])
    source_paths=[*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),PROTOCOL,REFERENCE,ROOT/'requirements.txt']
    source_hashes={artifact_label(p):sha(p) for p in source_paths}
    manifest={'started_utc':now(),'work_directory':str(WORK),'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib']},
              'input_hashes':{artifact_label(p):sha(p) for p in sorted((ROOT/'data/unified').glob('*.gz'))},
              'source_hashes':source_hashes,'jobs':jobs}
    def save():
        temp=OUT/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(OUT/'manifest.json')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(OUT/'matplotlib-cache'),MPLBACKEND='Agg')
    running={}
    try:
        while True:
            for name in list(running):
                process,log,start=running[name];code=process.poll()
                if code is not None:
                    log.close();jobs[name].update(status='complete' if code==0 else 'failed',returncode=code,completed_utc=now(),elapsed_seconds=round(time.monotonic()-start,2));del running[name]
                    print(jobs[name]['status'].upper()+': '+name,flush=True);save()
            failed=any(j['status']=='failed' for j in jobs.values())
            for name,job in jobs.items():
                if failed or len(running)>=args.jobs: break
                if job['status']=='pending' and all(jobs[d]['status']=='complete' for d in job['dependencies']):
                    log=(OUT/(name+'.log')).open('w');process=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    job.update(status='running',started_utc=now(),log=artifact_label(OUT/(name+'.log')))
                    running[name]=(process,log,time.monotonic());print('START: '+name,flush=True);save()
            if not running:
                if failed: raise SystemExit('Analysis failed; inspect the job log in '+str(OUT))
                if all(j['status']=='complete' for j in jobs.values()):
                    assert source_hashes=={artifact_label(p):sha(p) for p in source_paths}, 'Source changed during run'
                    check_inputs();manifest['completed_utc']=now();save();print(f'COMPLETE: {len(jobs)} jobs',flush=True);return
                raise RuntimeError('Unresolved dependencies')
            time.sleep(1)
    finally:
        for process,log,_ in running.values():
            process.terminate();process.wait();log.close()

if __name__=='__main__': main()

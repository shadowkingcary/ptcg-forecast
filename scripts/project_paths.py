# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Repository inputs and external working directories shared by all scripts."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def external(path):
    path=Path(path).expanduser().resolve()
    if path==ROOT or path.is_relative_to(ROOT):
        raise ValueError("Working files must be outside the repository: "+str(path))
    return path

WORK=external(os.environ.get('PTCG_WORK_DIR',ROOT.parent/'ptcg-forecast-work'))
RESULTS=WORK/'results'
FIGURES=WORK/'figures'
PROTOCOL=ROOT/'config/analysis_protocol.json'
REFERENCE=ROOT/'tests/fixtures/paper_checks.json'

def artifact_label(path):
    path=Path(path).resolve()
    if path.is_relative_to(ROOT): return str(path.relative_to(ROOT))
    if path.is_relative_to(WORK): return 'work/'+str(path.relative_to(WORK))
    return str(path)

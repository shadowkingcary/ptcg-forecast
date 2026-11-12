# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Shared API, cache and schema utilities for raw-data preparation.
Adapted from the project's October acquisition and historical joining scripts.
"""
import hashlib,json,re,time
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from project_paths import external
BASE='https://play.limitlesstcg.com/api'
FIELDS=['tournament_id','tournament_date','era','match_id','player_A','player_B','archetype_A','archetype_B','winner','round']
ROTATION_BOUNDARIES=['2024-03-21','2025-03-25','2026-03-28']
START,END,SPLIT='2023-12-30','2026-11-01','2026-10-01'

def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Client:
    def __init__(self):
        self.next_request = 0.0
        self.requests = 0

    def get(self, url):
        for attempt in range(7):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + 6.3
            try:
                request = Request(url, headers={"User-Agent": "ptcg-research/1.0"})
                with urlopen(request, timeout=30) as response:
                    data = json.load(response)
                    rate = response.headers.get("RateLimit", "")
                self.requests += 1
                remaining = re.search(r"\br=(\d+)", rate)
                reset = re.search(r"\bt=(\d+)", rate)
                if remaining and reset:
                    # Use the advertised allowance, then wait for its reset.
                    # Keep one request in reserve for other activity on this IP.
                    if int(remaining.group(1)) <= 1:
                        self.next_request = time.monotonic() + int(reset.group(1)) + 1
                        print(f"API allowance nearly used; waiting {int(reset.group(1)) + 1}s for reset", flush=True)
                    else:
                        self.next_request = time.monotonic() + 1.0
                return data, {"url": url, "retrieved_utc": utcnow(), "rate_limit": rate}
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 6:
                    raise
                retry = exc.headers.get("Retry-After", "")
                try:
                    delay = float(retry)
                except ValueError:
                    try:
                        delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                    except (TypeError, ValueError):
                        delay = min(60, 5 * 2 ** attempt)
                self.next_request = time.monotonic() + max(6.3, delay)
                print(f"HTTP {exc.code}; respecting retry delay {max(6.3, delay):.0f}s", flush=True)
            except (URLError, TimeoutError):
                if attempt == 6:
                    raise
                self.next_request = time.monotonic() + min(60, 5 * 2 ** attempt)
                print(f"Transient connection error, retry {attempt + 1}/6", flush=True)
        raise RuntimeError("Request retries exhausted")


def validate_array(data, endpoint):
    rows = data if isinstance(data, list) else data.get(endpoint) if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid {endpoint} response")
    return rows


def validate_tournament(item):
    if not isinstance(item,dict): raise ValueError('Tournament is not an object')
    tid=item.get('id')
    if not isinstance(tid,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',tid):
        raise ValueError('Invalid tournament ID')
    stamp=item.get('date','')
    parsed=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0:
        raise ValueError('Tournament dates must be UTC')
    if item.get('game')!='PTCG' or item.get('format')!='STANDARD':
        raise ValueError('Unexpected game or format')
    return item


def date_range(start,end):
    for value in (start,end): datetime.strptime(value,'%Y-%m-%d')
    if start>=end: raise ValueError('Start must precede end')


def read_endpoint(cache,tid,endpoint):
    path=cache/endpoint/(tid+'.json')
    provenance=path.with_suffix('.provenance.json')
    meta=json.loads(provenance.read_text())
    if meta.get('sha256')!=sha256(path): raise ValueError('Cache digest mismatch: '+str(path))
    return validate_array(json.loads(path.read_text()),endpoint)

# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Run download and raw-to-unified processing; working data stay outside the repo."""
import argparse,os
from pathlib import Path
from project_paths import WORK,external
from pipeline_utils import START,END,SPLIT
from download_tournaments import download
from build_unified import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['download','build','all'])
    p.add_argument('--work-dir',type=Path,default=WORK)
    p.add_argument('--start',default=START);p.add_argument('--end',default=END);p.add_argument('--split-date',default=SPLIT)
    p.add_argument('--reuse-listing',action='store_true');p.add_argument('--listing-file',type=Path)
    p.add_argument('--import-cache',type=Path);p.add_argument('--offline',action='store_true')
    p.add_argument('--preserve-labels-from',type=Path,action='append',default=[])
    p.add_argument('--identifier-map',type=Path);p.add_argument('--max-pages',type=int,default=10000)
    a=p.parse_args();work=external(a.work_dir);os.umask(0o077)
    if a.step in ('download','all'):
        download(work/'raw',a.start,a.end,reuse_listing=a.reuse_listing,listing_file=a.listing_file,
                 import_cache=a.import_cache,offline=a.offline,max_pages=a.max_pages)
    if a.step in ('build','all'):
        build(work/'raw',work/'rebuilt/unified',a.identifier_map or work/'private/identifier_map.json',a.split_date,a.preserve_labels_from)

if __name__=='__main__':main()

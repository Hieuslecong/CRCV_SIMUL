#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crcv_q1.gates import assess_file
p=argparse.ArgumentParser();p.add_argument("results_json");a=p.parse_args();r=assess_file(a.results_json);print(json.dumps(r,indent=2));raise SystemExit(0 if r["status"]=="Q1_READY" else 2)

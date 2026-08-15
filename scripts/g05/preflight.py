#!/usr/bin/env python3
"""Offline preflight for a VastAI G0.5 workspace; never downloads assets."""
import argparse, json, os, shutil
from pathlib import Path

def need(label, path, directory=True):
    p=Path(path).expanduser() if path else None
    ok=bool(p and (p.is_dir() if directory else p.is_file()))
    return {"name":label,"path":str(p) if p else None,"ok":ok}

ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); args=ap.parse_args()
checks=[need('dataset',os.environ.get('ROBODOJO_LEROBOT_V30_ROOT')),need('sidecar',os.environ.get('ROBODOJO_SIDECAR'),False),need('g05_root',os.environ.get('G05_ROOT')),need('base_assets',os.environ.get('G05_BASE_ASSETS'))]
checks += [{"name":"python3","ok":shutil.which('python3') is not None},{"name":"rsync","ok":shutil.which('rsync') is not None}]
m=Path(args.manifest); checks.append({"name":"subgoal_manifest","path":str(m),"ok":m.is_file()})
result={"checks":checks,"ok":all(x['ok'] for x in checks),"cwd":os.getcwd()}
print(json.dumps(result,indent=2,ensure_ascii=False))
raise SystemExit(0 if result['ok'] else 2)

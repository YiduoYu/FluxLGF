"""Run every locked batch sequentially, then add a third repeat on short/control families."""
import json,subprocess,sys
from pathlib import Path
from bench import batch
HERE=Path(__file__).resolve().parent
for stage in ['validate511','validate1023','validate2047']:
    subprocess.run([sys.executable,str(HERE/'bench.py'),stage],check=True)
lock=json.loads((HERE/'LOCK.json').read_text())
controls=[c for c in lock['cases']['validate511'] if c['fam'] not in ['circ','disk']]
batch([dict(**c,**s,tag=tag,rep=2) for c in controls for tag,s in lock['methods'].items()],
      'controls_extra.jsonl',seed=181116)

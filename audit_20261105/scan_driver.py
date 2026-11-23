import subprocess,os,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
configs=[(t,a,4,'HMIS') for t in (.25,.5,.7) for a in (0,1,2)]
configs +=[(.5,1,2,'HMIS'),(.5,1,8,'HMIS'),(.5,1,4,'PMIS'),(.5,1,4,'bt-fd')]
jobs=[(m,n,rep,c) for m,n in [(32,1),(32,10),(64,1)] for rep in (0,1) for c in configs]
random.Random(20261105).shuffle(jobs)
with (HERE/'evidence/amg_scan.jsonl').open('w') as fh:
 for i,(m,n,rep,c) in enumerate(jobs):
  args=[str(m),str(n),*[str(x) for x in c],'ir']
  p=subprocess.run(['/opt/anaconda3/envs/hypre_env/bin/python',str(HERE/'probe_amg.py'),*args],env=env,text=True,capture_output=True,timeout=180)
  try: row=json.loads(p.stdout.splitlines()[-1]);row['repeat']=rep
  except Exception: row=dict(m=m,n=n,repeat=rep,config=c,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  fh.write(json.dumps(row)+'\n');fh.flush()
  print(i+1,len(jobs),m,n,c,row.get('total_s'),row.get('success'),flush=True)

# Timing diagnostic after interrupted execution, 2026-11-17

The first execution session stopped after 132/200 records and was unavailable on
the next user turn, about 11 hours later. The original locked runner resumes the
missing job identities without any algorithm or parameter change. All records stay.

To check session drift, after the original 200 records finish, run Pmax=8 twice on
ALL eight cases in randomized order seed 17117. Save these 16 calibration runs in
`anchors.jsonl`, never pool them into the formal median table or select replacements.
Compare their per-case median wall and CPU time against Pmax=8 measurements among
the first 132 records. If a case lacks a first-session Pmax=8 record, say so.

These anchors do not repair or statistically remove drift. They flag whether small
differences against direct spectral methods deserve weaker interpretation. Report
their range alongside CPU/wall consistency, and preserve the 200-record estimates.
No adjustment factor is fitted to the candidates. Run sequentially, not alongside
diagnostics or any other benchmark.

Before running any anchors, add this conditional replication rule: if ANY available
anchor wall ratio is outside [0.90,1.10], run a within-session paired timing check
for ALL eight cases. Compare fixed96 weighted lifting against the per-case fastest
direct rank from the original 200-record table, two fresh repeats each (32 runs).
Use randomized case/repetition blocks and randomized within-pair order (seed 17118).
This chooses an already tested comparator, not new algorithm parameters, and does
not tune or replace original results. Save separately in confirmation.jsonl and
report the paired confirmation alongside the original result. Its purpose is timing
replication on known cases, NOT a new holdout/generalization test.

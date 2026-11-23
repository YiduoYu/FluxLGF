# Timing replication addendum (before extra runs)

After completing validate511, some controls that did not activate a correction showed
wall-time ratios up to 1.161, while their CPU-time ratios were about 1.059. Individual
runs varied noticeably even though repeated iteration counts were identical.

The protocol permits at least three repeats for short controls if needed. Extend the
already planned third repeat on six extra families uniformly to ALL ten moderate-
anisotropy circ/disk control cases, on both grids, with all four methods. Do not change
solver code, lock, parameters, cases, or any previous result. Store the 40 extra rows
in controls_moderate_extra.jsonl; identity rep=2. This is timing replication after
observing variance, not new independent problems or a post-hoc algorithm improvement.

The full set then has 304 measurements: 14 hard cases with two repeats per method,
14 controls and two topology cases with three repeats per method. Also report the
original two-repeat control statistic as a sensitivity check, so the extension cannot
silently replace an unfavorable original timing result. Extra repetitions do not
establish statistical significance or remove the need for a quiet benchmark host.

#!/usr/bin/env bash
# Commands that produced the runs reported in Appendix A.4 of
# "Learning Lax Pairs: Revisiting the Classical Paradigm".
# Output goes to ./output (override with SILO_OUT=/path).
set -e
cd "$(dirname "$0")"
python silo/euler_top.py 60                 # protocols A,B,C seeds 0-59; D seeds 0-19
python silo/schrodinger.py 50 det pinv      # seeds 0-49 per barrier
python silo/burgers.py 60                   # seeds 0-59
python silo/shallow_water.py base:28        # seeds 0-27, penalty on
python silo/shallow_water.py base_nopenalty # seeds 0-9, penalty off
python silo/shallow_water.py enl3:2         # window (3,3,3,3), seeds 0-1 (about 25 min each)
python silo/summarize.py "${SILO_OUT:-output}"

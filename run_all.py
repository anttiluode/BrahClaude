"""Run every gate and redraw the figures (CPU, ~1 minute)."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
sys.path.insert(0, EXP)
os.chdir(EXP)
for name in ["g0_identities", "g1_identify", "g2_silent_memory", "g3_write_for_later",
             "g4_multiplex", "figures"]:
    print(f"\n=== {name} ===")
    runpy.run_path(os.path.join(EXP, name + ".py"), run_name="__main__")

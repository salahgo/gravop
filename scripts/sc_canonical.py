"""Supported entry point for the complete converged-clock method sequence.

The legacy sc_experiment module remains frozen for artifact compatibility.
"""
from reproduce import run_target
import argparse

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='rebuild in a prepared, separate checkout')
    a=p.parse_args()
    run_target('methods',execute=a.execute)

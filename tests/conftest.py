"""Put src/ on sys.path so the tests can import the modules by their flat names.

The solver modules import each other flatly (``from tt_dmrg import ...``) because
the "Glauber TT" paper bundle ships byte-identical copies of them in a single
directory and has to stay runnable on its own.  Tests live outside src/, so they
need that directory on the path explicitly.
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
sys.path.insert(0, os.path.abspath(SRC))

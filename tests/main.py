"""
This file runs some basic testing scripts for regression testing.
"""
import sys
sys.path.append('..')
import numpy as np
import fb_compute as fb
from Multinoulli import Multinoulli
from HMM import HMM
from States import States
import fb_compute_test
import VB_test
import SVI_test
import HMM_test

print("***************TESTING*********************")
correct = 0
total = 4

correct += HMM_test.run()

correct += fb_compute_test.run()

correct += VB_test.run()

correct += SVI_test.run()

print("FINAL: passed " + str(correct) + "/" + str(total) + " tests")
print("**************DONE TESTING*****************")

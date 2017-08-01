"""
This file runs some basic testing scripts for regression testing.
"""
# internals
import States_test
import VB_test
import SVI_test
import HMM_test

print("***************TESTING*********************")
correct = 0
total = 4

correct += HMM_test.run()

correct += States_test.run()

correct += VB_test.run()

correct += SVI_test.run()

print("FINAL: passed " + str(correct) + "/" + str(total) + " tests")
print("**************DONE TESTING*****************")

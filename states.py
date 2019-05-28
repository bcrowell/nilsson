"""
Utility functions for dealing with states, e.g., describing them with strings.
"""

def describe_omega_parity(omega,parity):
  s = str(omega)+"/2"
  if parity==0:
    s = s+"+"
  else:
    s = s+"-"
  return s

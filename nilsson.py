#!/usr/bin/python3

"""
An implementation of the Nilsson model of nuclear structure for deformed nuclei.
"""

import numpy

def main():
  states = enumerate_states(5,7,1) # 7/2+

def enumerate_states(n_max,omega,parity):
  """
  Returns a hash whose keys are 4-tuples containing Nilsson quantum numbers, and whose values are indices 0, 1, ...
  """
  # Integer spins are represented as themselves. Half-integer spins are represented by 2 times themselves.
  # parity = 0 for even parity, 1 for odd
  # omega = 2*Omega, should be positive
  states = {}
  i = 0
  for n in range(parity,n_max+1,2):
    for l in range(0,n+1):
      for ml in range(-l,l+1): # m_l
        for ms in range(-1,1+1,2): # two times m_s
          if 2*ml+ms==omega:
            print("n,l,ml,ms=",n,l,ml,ms)
            states[(n,l,ml,ms)] = i
            i = i+1
  return states


main()

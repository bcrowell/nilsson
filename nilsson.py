#!/usr/bin/python3

"""
An implementation of the Nilsson model of nuclear structure for deformed nuclei.
"""

import numpy

def main():
  n_max = 5
  omega = 7 # 7/2-
  parity = 1
  space = (n_max,omega,parity)
  index = enumerate_states(space) # dictionary mapping from Nilsson quantum numbers to array indices
  states = {v: k for k, v in index.items()} # inverse dictionary, maps from index to quantum numbers
  print(index)
  print(states)
  n_states = len(states)
  default_pars = {'kappa':0.06,'mu':0.5,'delta':0.0}
  user_pars = {'delta':0.2}
  pars = {**default_pars,**user_pars} # merge dictionaries, second one overriding first
  ham = hamiltonian(space,pars,index,states)
  evals,evecs = eigen(ham)
  for i in range(n_states):
    print(evals[i])

def eigen(a):
  eigenvalues, eigenvectors = numpy.linalg.eig(a) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
  # Now sort them, https://stackoverflow.com/a/50562995/1142217
  idx = numpy.argsort(eigenvalues)
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:,idx]
  return (eigenvalues, eigenvectors)

def hamiltonian(space,pars,index,states):
  """
  Returns the Hamiltonian matrix, with matrix elements in units of omega0.
  """
  n_max,omega,parity = space
  kappa = pars['kappa']
  mu =    pars['mu']
  delta = pars['delta']
  n_states = len(states)
  ham = numpy.zeros(shape=(n_states,n_states)) # hamiltonian
  for i in range(n_states):
    n,l,ml,ms = states[i]
    ham[i,i] = ham[i,i]+1.5+n
  return ham

def enumerate_states(space):
  """
  Returns a hash whose keys are 4-tuples containing Nilsson quantum numbers, and whose values are indices 0, 1, ...
  """
  n_max,omega,parity = space
  # Integer spins are represented as themselves. Half-integer spins are represented by 2 times themselves.
  # parity = 0 for even parity, 1 for odd
  # omega = 2*Omega, should be positive
  index = {}
  i = 0
  for n in range(parity,n_max+1,2):
    for l in range(0,n+1):
      for ml in range(-l,l+1): # m_l
        for ms in range(-1,1+1,2): # two times m_s
          if 2*ml+ms==omega:
            index[(n,l,ml,ms)] = i
            i = i+1
  return index


main()

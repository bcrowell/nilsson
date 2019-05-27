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
  c1 = 2.0*kappa
  c2 = mu*kappa
  for i in range(n_states):
    n,l,ml,ms = states[i]
    ham[i,i] = ham[i,i] + 1.5+n -c2*(l*(l+1)-0.5*n*(n+3)) -c1*ms*ml   # 3/2+N and l^2 terms, plus diagonal part of spin-orbit term
    # off-diagonal part of spin-orbit:
    for j in range(n_states):
      n2,l2,ml2,ms2 = states[j]
      if n2!=n:
        continue # Check this...? I think this is right, should vanish unless the N's are equal, because radial w.f. are orthogonal.
      if l2!=l:
        continue
      for sign in range(-1,1+1,2):
        if ml+sign==ml2 and ms==-sign and ms2==sign:
          ham[i,j] = ham[i,j]+0.5*math.sqrt((l-sign*ml)*(l+sign*ml+1))
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

def eigen(a):
  eigenvalues, eigenvectors = numpy.linalg.eig(a) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
  # Now sort them, https://stackoverflow.com/a/50562995/1142217
  idx = numpy.argsort(eigenvalues)
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:,idx]
  return (eigenvalues, eigenvectors)

#------------------------------------------------

main()

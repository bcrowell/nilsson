"""
High-level routines.
"""

import util,hamiltonian

def do_nilsson(n_max,omega,parity,user_pars):
  """
  Find the energies and eigenstates for the Nilsson model, with the given Omega(=Jz) and parity (0=even, 1=odd).
  Energies are reported in units of omega00=(42 MeV)/A^(1/3).
  User_pars is a hash such as {'kappa':0.06,'mu':0.5,'delta':0.2}. Sane defaults are provided for all parameters for
  testing purposes, and these parameters are also given back in the returned hash.
  Returns a hash with keys n_states,states,index,evals,evecs,ham.
  """
  if omega%2==0:
    raise Excaption("even value of Omega in do_nilsson")
  space = (n_max,omega,parity)
  index = hamiltonian.enumerate_states(space) # dictionary mapping from Nilsson quantum numbers to array indices
  states = util.dict_to_tuple(util.invert_dict(index)) # inverse of index, maps from index to quantum numbers
  n_states = len(states)
  default_pars = {'kappa':0.06,'mu':0.5,'delta':0.0}
  pars = util.merge_dicts(default_pars,user_pars)
  ham = hamiltonian.hamiltonian(space,pars,index,states)
  evals,evecs = hamiltonian.eigen(ham)
  return util.merge_dicts(pars,{'n_states':n_states,'states':states,'index':index,'evals':evals,'evecs':evecs,'ham':ham})

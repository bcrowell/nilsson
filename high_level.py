"""
High-level routines.
"""

import util,hamiltonian

def do_nilsson(n_max,omega,parity,user_pars):
  """
  Find the energies and eigenstates for the Nilsson model, with the given Omega(=Jz) and parity (0=even, 1=odd).
  Energies are reported in units of omega00, and should be rescaled by omega0 and also by (42 MeV)/A^(1/3).
  User_pars is a hash such as {'kappa':0.06,'mu':0.5,'delta':0.2}. Sane defaults are provided for all parameters for
  testing purposes, and these parameters are also given back in the returned hash.
  Returns a hash with keys n_states,states,index,omega0,evals,evecs,ham.
  """
  space = (n_max,omega,parity)
  index = hamiltonian.enumerate_states(space) # dictionary mapping from Nilsson quantum numbers to array indices
  states = util.dict_to_tuple(util.invert_dict(index)) # inverse of index, maps from index to quantum numbers
  n_states = len(states)
  default_pars = {'kappa':0.06,'mu':0.5,'delta':0.0}
  pars = util.merge_dicts(default_pars,user_pars)
  ham = hamiltonian.hamiltonian(space,pars,index,states)
  evals,evecs = hamiltonian.eigen(ham)
  omega0 = hamiltonian.delta_to_omega0(pars['delta'])
  return util.merge_dicts(pars,{'n_states':n_states,'states':states,'index':index,'omega0':omega0,'evals':evals,'evecs':evecs,'ham':ham})

#!/usr/bin/python3

"""
An implementation of the Nilsson model of nuclear structure for deformed nuclei.
"""

import math,numpy,sympy
from sympy.physics.quantum.cg import CG

def main():
  do_nilsson()

def do_nilsson():
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
  Returns the Hamiltonian matrix, with matrix elements in units of omega00 (not omega0, which has a second-order dependence
  on deformation).
  """
  n_max,omega,parity = space
  kappa = pars['kappa']
  mu =    pars['mu']
  delta = pars['delta']
  n_states = len(states)
  ham = numpy.zeros(shape=(n_states,n_states)) # hamiltonian
  c1 = 2.0*kappa
  c2 = mu*kappa
  #     We apply this at the end to the entire hamiltonian.
  # 3/2+N, l^2, and spin-orbit terms:
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
  # deformation term:
  def_con = -(4.0/3.0)*math.sqrt(math.pi/5.0)*delta # proportionality constant for r^2Y20 term
  for i in range(n_states):
    n,l,ml,ms = states[i]
    for j in range(n_states):
      n2,l2,ml2,ms2 = states[j]
      # matrix elements of deformation term
      if j<=i:
        z = r2_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2)*y20_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2)
        # ... I assume multiplying these is the right thing to do, since the integrals are separable,
        ham[i,j] = ham[i,j] + z
        ham[j,i] = ham[j,i] + z
  omega0 = (1-(4.0/3.0)*delta**2-(16.0/27.0)*delta**3)**(-1.0/6.0)
  # ... rescaled from omega00=1 for volume conservation
  for i in range(n_states):
    for j in range(n_states):
      ham[i,j] = ham[i,j]*omega0
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

def y20_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2):
  """
  Compute the matrix element <l2 ml2 | Y20 | l ml>.
  """
  # https://physics.stackexchange.com/questions/10039/integral-of-the-product-of-three-spherical-harmonics
  if not (ml==ml2 and ms==ms2 and abs(l-l2)<=2 and (l-l2)%2==0 and abs(n-n2)==2):
    return 0.0
  # Beyond this point, we don't look at ms or ms2 anymore, so all spins are integers.
  x = math.sqrt((5.0/(4.0*math.pi)) * ((2*l+1)/(2*l2+1)))
  if l-l2!=0:
    x = -x
  x = x*clebsch(l,2,ml,0,l2,ml2)*clebsch(l,2,0,0,l2,0)
  return x

def r2_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2):
  """
  Compute the matrix element <n2 l2 | r^2 | n l>.
  """
  # nuclear.fis.ucm.es/PDFN/documentos/Nilsson_Doct.pdf
  if ml!=ml2 or ms!=ms2 or abs(n-n2)!=0:
    return 0.0
  if n==n2:
    return n+1.5 # special case for efficiency
  # Beyond this point, we don't look at ms or ms2 anymore, so all spins are integers. Only p is a half-integer. mu, nu, d, sigma are integera
  p = 0.5*(l+l2+3)
  mu = p-l2-0.5
  nu = p-l-0.5
  d = 0.5*(n-l)+1 # UCM's lowercase n
  d2 = 0.5*(n2-l2)+1
  sum = 0.0
  for sigma in range(max(d2-mu-1,d-nu-1),min(d-1,d2-1)+1): # guess range based on criterion that factorials should be >=0
    ln_term = ln_gamma(p+sigma+1)-(ln_fac(sigma)+ln_fac(d2-1-sigma)+ln_fac(d-1-sigma)+ln_fac(sigma+mu-d2+1)+ln_fac(sigma+nu-d+1))
    sum = sum + math.exp(ln_term)
  ln_stuff = ln_fac(d2-1)+ln_fac(d-1)-(ln_gamma(n2+l2+0.5)+ln_gamma(n+l+0.5))
  ln_stuff2 = ln_fac(mu)+ln_fac(nu)
  result = sum*math.exp(0.5*ln_stuff+ln_stuff2)
  if (d+d2)%2!=0:
    result = -result
  # ECM has a (unitful?) factor of b^2, but doesn't include that factor in their sample expressions for <N|...|N+2>.
  # They define b as sqrt(hbar/m*omega0), so when we compute energies in units of hbar*omega0, this should not be an issue.
  return result

def ln_fac(n):
  return scipy.special.gammaln(n+1)

def ln_gamma(x):
  # make a separate function so we can memoize it
  return scipy.special.gammaln(x)

def clebsch(l1,l2,ml1,ml2,l3,ml3):
  """
  Computes < l1 l2 ml1 ml2 | l3 ml3>, where all spins are integers (no half-integer spins allowed).
  This is just a convenience function.
  """
  return clebsch2(l1*2,ml1*2,l2*2,ml2*2,l3*2,ml3*2)

def eigen(a):
  """
  Compute sorted eigenvectors and eigenvalues of the matrix a.
  """
  eigenvalues, eigenvectors = numpy.linalg.eig(a) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
  # Now sort them, https://stackoverflow.com/a/50562995/1142217
  idx = numpy.argsort(eigenvalues)
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:,idx]
  return (eigenvalues, eigenvectors)

def clebsch2(j1,m1,j2,m2,j3,m3):
  """
  Computes <j1 m1 j2 m2 | j3 m3>, where all spins are given as double their values (contrary to the usual convention in this code).
  """
  # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
  # https://mattpap.github.io/scipy-2011-tutorial/html/numerics.html
  # This is kind of silly, using a symbolic math package to compute these things numerically, but I couldn't find a convenient
  # numerical routine for this that was licensed appropriately and packaged for ubuntu.
  # Performance is actually fine, because this is memoized. We take a ~1 second hit in startup time just from loading sympy.
  return CG(sympy.S(j1)/2,sympy.S(m1)/2,sympy.S(j2)/2,sympy.S(m2)/2,sympy.S(j3)/2,sympy.S(m3)/2).doit().evalf()

class Memoize: 
  # https://stackoverflow.com/a/1988826/1142217
  def __init__(self, f):
    self.f = f
    self.memo = {}
  def __call__(self, *args):
    if not args in self.memo:
      self.memo[args] = self.f(*args)
    return self.memo[args]

clebsch = Memoize(clebsch)
y20_matrix_element = Memoize(y20_matrix_element)
r2_matrix_element = Memoize(r2_matrix_element)
ln_fac = Memoize(ln_fac)
ln_gamma = Memoize(ln_gamma)
  
#------------------------------------------------

main()

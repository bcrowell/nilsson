"""
Functions for computing the Nilsson Hamiltonian. As a convention throughout this code (except where noted),
integer spins are stored as their actual values, while half-integer spins are stored
as double their actual values.
"""

#-----------------------------------------------------------------------------------------------

import math,numpy,sympy,scipy.special
from sympy.physics.quantum.cg import CG

import util
from util import Memoize

#-----------------------------------------------------------------------------------------------

def hamiltonian(space,pars,index,states):
  """
  Returns the Hamiltonian matrix, with matrix elements in units of omega00.
  When running a series of calculations with different deformations, some of the more computationally expensive
  parts of this calculation will automatically be memoized.
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
    ham[i,i] = ham[i,i] + 1.5+n -c2*(l*(l+1)-0.5*n*(n+3)) -c1*ml*(0.5*ms)   # 3/2+N and l^2 terms, plus diagonal part of spin-orbit term
    # off-diagonal part of spin-orbit:
    for j in range(n_states):
      if j==i:
        continue # already did diagonal elements
      n2,l2,ml2,ms2 = states[j]
      if n2!=n:
        continue # Check this...? I think this is right, should vanish unless the N's are equal, because radial w.f. are orthogonal.
      if l2!=l:
        continue
      for sign in range(-1,1+1,2):
        if ml+sign==ml2 and ms==sign and ms2==-sign:
          ham[i,j] = ham[i,j]-c1*0.5*math.sqrt((l-sign*ml)*(l+sign*ml+1))
  # deformation term, proportional to r^2 Y20
  omega0 = delta_to_omega0(delta)
  def_con = -omega0*(4.0/3.0)*math.sqrt(math.pi/5.0)*delta # proportionality constant for this term
  d = deformation_ham(space,states)
  for i in range(n_states):
    for j in range(n_states):
      ham[i,j] = ham[i,j]+d[i,j]*def_con
  return ham

def delta_to_omega0(delta):
  """
  Find value of omega0, rescaled from omega00=1 for volume conservation.
  """
  return (1-(4.0/3.0)*delta**2-(16.0/27.0)*delta**3)**(-1.0/6.0)

@Memoize
def deformation_ham(space,states):
  """
  Computes the term in the Hamiltonian that represents the deformation, not including scalar factors. This is a separate function so
  that it can be memoized. When running a series of calculations with different deformations, this doesn't need to be recomputed.
  """
  n_states = len(states)
  ham = numpy.zeros(shape=(n_states,n_states))
  for i in range(n_states):
    n,l,ml,ms = states[i]
    for j in range(n_states):
      n2,l2,ml2,ms2 = states[j]
      # matrix elements of deformation term
      if j<=i and abs(l-l2)<=2:
        z = r2_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2)*y20_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2)
        # ... I assume multiplying these is the right thing to do, since the integrals are separable,
        ham[i,j] = ham[i,j] + z
        if i!=j:
          ham[j,i] = ham[j,i] + z
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
      if (n-l)%2!=0: # parity
        continue
      for ml in range(-l,l+1): # m_l
        for ms in range(-1,1+1,2): # two times m_s
          if 2*ml+ms==omega:
            index[(n,l,ml,ms)] = i
            i = i+1
  return index

@Memoize
def y20_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2):
  """
  Compute the matrix element <l2 ml2 | Y20 | l ml>.
  """
  # https://physics.stackexchange.com/questions/10039/integral-of-the-product-of-three-spherical-harmonics
  if not (ml==ml2 and ms==ms2 and abs(l-l2)<=2 and (l-l2)%2==0):
    return 0.0
  # Beyond this point, we don't look at ms or ms2 anymore, so all spins are integers.
  x = math.sqrt((5.0/(4.0*math.pi)) * ((2*l+1)/(2*l2+1)))
  if l-l2!=0:
    x = -x
  x = x*clebsch(l,2,ml,0,l2,ml2)*clebsch(l,2,0,0,l2,0)
  return x

@Memoize
def r2_matrix_element(n,l,ml,ms,n2,l2,ml2,ms2):
  """
  Compute the matrix element <n2 l2 | r^2 | n l>.
  """
  # nuclear.fis.ucm.es/PDFN/documentos/Nilsson_Doct.pdf
  if ml!=ml2 or ms!=ms2 or (n-n2)%2!=0 or (n-n2)>2:
    return 0.0
  # If d were a half-integer, then we couldn't have both sigma! and (d-1-sigma)! make sense. This seems to tell me that
  # if N-l isn't even, the matrix element must vanish. This sort of makes sense given UCM's notation lowercase n for my d, probably
  # similar to hydrogen atom's n.
  if (n-l)%2!=0 or (n2-l2)%2!=0:
    return 0.0
  # Beyond this point, we don't look at ms or ms2 anymore, so all spins are integers. Only p is half-integer. d, d2, mu, nu, sigma are integers.
  p = 0.5*(l+l2+3)
  mu = util.to_int(p-l2-0.5)
  nu = util.to_int(p-l-0.5)
  d = util.to_int(0.5*(n-l)+1) # UCM's lowercase n
  d2 = util.to_int(0.5*(n2-l2)+1)
  sum = 0.0
  for sigma in range(max(0,d2-mu-1,d-nu-1),min(d-1,d2-1)+1): # guess range based on criterion that all 5 inputs to factorials should be >=0
    ln_term = ln_gamma(p+sigma+1)-(ln_fac(sigma)+ln_fac(d2-1-sigma)+ln_fac(d-1-sigma)+ln_fac(sigma+mu-d2+1)+ln_fac(sigma+nu-d+1))
    term = math.exp(ln_term)
    sum = sum + term
  ln_stuff = ln_fac(d2-1)+ln_fac(d-1)-(ln_gamma(d2+l2+0.5)+ln_gamma(d+l+0.5))
  if mu<0 or nu<0:
    print("negative mu or nu in r2_matrix_element -- this shoudln't happen")
    print("p,d,mu,nu=",p,d,mu,nu)
    print("n,l,ml,ms,    n2,l2,ml2,ms2=",n,l,ml,ms,"    ",n2,l2,ml2,ms2)
  ln_stuff2 = ln_fac(mu)+ln_fac(nu)
  result = sum*math.exp(0.5*ln_stuff+ln_stuff2)
  if (d+d2)%2!=0:
    result = -result
    # ... Note that this is the only place where a sign can occur. The gamma function inside the sum is always
    #     positive, because both sigma and p are nonnegative.
  # ECM has a (unitful?) factor of b^2, but doesn't include that factor in their sample expressions for <N|...|N+2>.
  # They define b as sqrt(hbar/m*omega0), so when we compute energies in units of hbar*omega0, this should not be an issue.
  return result

@Memoize
def ln_fac(n):
  # The following checks should only cause exceptions if there's an error in my algorithms. However, they have
  # little run-time cost because this gets memoized.
  if n<0:
    raise Exception('negative input in ln_fac')
  if not util.has_integer_value(n): 
    raise Exception('non-integer input in ln_fac')
  return scipy.special.gammaln(n+1)

@Memoize
def ln_gamma(x):
  # make a separate function so we can memoize it
  return scipy.special.gammaln(x)

@Memoize
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
  # Looking for a better alternative: https://scicomp.stackexchange.com/questions/32744/plug-and-go-clebsch-gordan-computation-in-python
  return CG(sympy.S(j1)/2,sympy.S(m1)/2,sympy.S(j2)/2,sympy.S(m2)/2,sympy.S(j3)/2,sympy.S(m3)/2).doit().evalf()

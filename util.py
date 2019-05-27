"""
Low-level utility functions not having to do with physics.
"""

class Memoize: 
  # https://stackoverflow.com/a/1988826/1142217
  def __init__(self, f):
    self.f = f
    self.memo = {}
  def __call__(self, *args):
    if not args in self.memo:
      self.memo[args] = self.f(*args)
    return self.memo[args]

def invert_dict(d):
  return {v: k for k, v in d.items()}

def dict_to_tuple(d): # convert a dictionary with integer keys to a tuple so it can be hashed and used in memoization
  a = []
  for i in d:
    a.append(d[i])
  return tuple(a)
  
def merge_dicts(f,g):
  return {**f,**g} # merge dictionaries, second one overriding first

def has_integer_value(x):
  """
  Tells whether an int or float has a value that is an integer
  """
  # https://stackoverflow.com/a/9266979/1142217
  return isinstance(x, int) or x.is_integer()

def to_int(x):
  if has_integer_value(x):
    return int(x)
  else:
    raise Exception('non-integer value in to_int(): ',x)

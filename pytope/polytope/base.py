import numpy as np
import cdd  # pycddlib -- for vertex enumeration from H-representation
from scipy.spatial import ConvexHull  # for finding A, b from V-representation
from scipy.optimize import linprog  # for support, projection, and more
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import math
import time

class EmptyPolytope():
    def __init__(self, n):
        if not isinstance(n, int):
            raise ValueError('Dimension of Polytope must be an integer.')
        self._n = n

    @property
    def n(self):
        return self._n

    def __add__(self, y):
        if isinstance(y, BasePolytope):
            if y.n != self.n:
                raise ValueError('Cannot add polytopes of different '
                    'dimensions: {} != {}'.format(self.n, y.n))

            return y

        elif isinstance(y, np.ndarray):
            if len(y.shape) > 1 or y.shape[0] != self.n:
                raise ValueError('Cannot add polytope of dimension {} and '
                    'numpy ndarray of shape {}. Can only add ')
        else:
            return NotImplemented

    def __radd__(self, y):
        self.__add__(y)

class VertexPolytope():
    def __init__(self, V, locked=False):
        if not isinstance(V, np.ndarray) or len(V.shape) > 2:
            raise ValueError('Vertices for VertexPolytope must be specified '
                'as a 1 or 2-dimensional numpy ndarray.')

        self._V = np.atleast_2d(V)

    @property
    def V(self):
        return self.V

    @property
    def n(self):
        return self.V.shape[1]

    def __add__(self, y):
        pass

class HalfspacePolytope():
    def __init__(H=None, A=None, b=None, locked=False):
        pass

    @property
    def A(self):
        pass

    @property
    def b(self):
        pass

    @property
    def H(self):
        pass

class Polytope(HalfspacePolytope, VertexPolytope):
    def __init__(Hpoly, Vpoly):
        pass

class Polytope:

  def __init__(self, *args, **kwargs):

    # Set n = 0 if constructed as Polytope() or without n as a kwarg:
    n = 0 if not (args or kwargs) else kwargs.get('n', 0)

    # Check if the constructor was called to create an empty Polytope in R^n --
    # Polytope(n=n). This is only allowed if n is the only argument.
    if 'n' in kwargs and (args or len(kwargs) > 1):
      raise ValueError('Cannot set dimension n with other arguments') 

    # Set some defaults, potentially changed below:
    self.n = n
    self.in_H_rep = False
    self.in_V_rep = False
    self._A = np.empty((0, n))
    self._b = np.empty((0, 1))
    self._V = np.empty((0, n))
    self._dim = None  # unknown

    # Check how the constructor was called. TODO: account for V=None, V=[],
    # or similar
    V_or_R_passed = len(args) == 1 or any(kw in kwargs for kw in ('V', 'R'))
    A_and_b_passed = len(args) == 2 or all(k in kwargs for k in ('A', 'b'))
    lb_or_ub_passed = any(kw in kwargs for kw in ('lb', 'ub'))

    if V_or_R_passed and A_and_b_passed:
      raise ValueError('Cannot specify V in addition to A and b')

    if (V_or_R_passed or A_and_b_passed) and lb_or_ub_passed:
      raise ValueError('Cannot specify bounds in addition to V, R, A, or b')

    if ('A' in kwargs) ^ ('b' in kwargs):  # XOR
      raise ValueError('Cannot pass just one of A and b as keywords')

    # Parse V if passed as the only positional argument or as a keyword
    if V_or_R_passed:
      V = kwargs.get('V')  # None if not
      if len(args) == 1:  # V is the only positional argument
        # Prevent passing vertex lists as first positional argument and as
        # keyword, as in P = Polytope(V_list1, V=V_list2):
        if V is not None:
          raise ValueError('V cannot be passed as the first argument and as a '
                           'keyword')
        V = args[0]  # The first positional argument is V
      # Parse R if passed as keyword
      if 'R' in kwargs:
        raise NotImplementedError('Support for rays currently not implemented')
      self._set_V(V)

    # Parse A and b if passed as first two positional arguments or as keywords
    if A_and_b_passed:
      A = kwargs.get('A')  # None if not
      b = kwargs.get('b')  # None if not
      # Prevent passing A or b in both args and kwargs:
      if len(args) == 2:  # A and b passed in args
        if A is not None or b is not None:  # A or b passed in kwargs
          raise ValueError(('A (or b) cannot be passed as the first (or second)'
                            ' argument and as a keyword'))
        A, b = args[:2]  # The first two positional arguments are A and b
      self._set_Ab(A, b)

    if lb_or_ub_passed:
      # Parse lower and upper bounds. Defaults to [], rather than None,
      # if key is not in kwargs (cleaner below).
      lb = np.atleast_1d(np.squeeze(kwargs.get('lb', [])))
      ub = np.atleast_1d(np.squeeze(kwargs.get('ub', [])))
      if (lb > ub).any():
        raise ValueError('No lower bound can be greater than an upper bound')
      self._set_Ab_from_bounds(lb, ub)  # sets A, b, n, and in_H_rep (to True)

  # To enable linear mapping (a numpy ndarray M multiplies a polytope P: M * P)
  # with the * operator, set __array_ufunc__ = None so that the result is not
  # M.__mul__(P), which is an ndarray of scalar multiples of P: {m_ij * P}. With
  # __array_ufunc__ set to None here, the result of M * P is P.__rmul__(M),
  # which calls P.linear_map(M). See
  # https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
  # An alternative is to set __array_priority__ = 1000 or some other high value.
  __array_ufunc__ = None

  @property
  def A(self):
    return self._get_A()

  @A.setter
  def A(self, A):
    return self._set_A(A)

  def _get_A(self):
    if not self.in_H_rep and self.in_V_rep:
      self.determine_H_rep()
    return self._A

  def _set_A(self, A):  # careful if setting A independent of b
    self._A = np.array(A, ndmin=2, dtype=float)  # prevents shape (n,) for m = 1

  @property
  def b(self):
    return self._get_b()

  @b.setter
  def b(self, b):
    return self._set_b(b)

  def _get_b(self):
    if not self.in_H_rep and self.in_V_rep:
      self.determine_H_rep()
    return self._b

  def _set_b(self, b):  # Careful when setting b indep of A (use for scaling P)
    # Ensure np.hstack((A, b)) works by having b be a column vector
    self._b = np.atleast_2d(np.squeeze(b).astype(float)).T

  @property
  def H(self):  # the matrix H = [A b]
    return self._get_H_mat()

  def _get_H_mat(self):
    return np.hstack((self.A, self.b))

  def get_H_rep(self):  # the pair -- or tuple -- (A, b)
    return (self.A, self.b)

  def _set_Ab(self, A, b):
    A = np.array(A, ndmin=2)  # ndmin=2 prevents shape (n,) when m = 1
    b = np.atleast_2d(np.squeeze(b).astype(float)).T  # overlaps with _set_b(b)
    m, n = A.shape
    if b.shape[0] != m:
      raise ValueError(f'A has {m} rows; b has {b.shape[0]} rows!')
    # For rows with b = +- inf: set a_i = 0' and b_i = +- 1 (indicates which
    # direction the constraint was unbounded -- not important)
    inf_rows = np.squeeze(np.isinf(b))  # True also for -np.inf
    if inf_rows.any():
      A[inf_rows, :] = 0
      b[inf_rows] = 1 * np.sign(b[inf_rows])
    self._set_A(A)
    self._set_b(b)
    self.n = n
    if A.shape[0] and b.shape[0]:  # in_H_rep if A and b are not empty arrays
      self.in_H_rep = True

  def _set_Ab_from_bounds(self, lb, ub):
    A_bound = []
    b_bound = []
    n = np.max((lb.size, ub.size))  # 0 for bound not specified
    # Test size of lb instead of "if lb" since the bool value of
    # np.array(None) is False whereas np.array([]) is ambiguous. Since
    # np.array(None).size is 1 and np.array([]).size is 0 (default, set
    # above), lb.size > 0 implies lb was a kwarg.
    if lb.size > 0:
      if not lb.size == n:
        raise ValueError((f'Dimension of lower bound lb is {lb.size}; '
                          f'should be {n}'))
      A_bound.extend(-np.eye(n))
      b_bound.extend(-lb)
    if ub.size > 0:
      if not ub.size == n:
        raise ValueError((f'Dimension of upper bound ub is f{ub.size}; '
                          f'should be {n}'))
      A_bound.extend(np.eye(n))
      b_bound.extend(ub)
      self._set_Ab(A_bound, b_bound)  # sets n and in_H_rep to True

  @property
  def V(self):
    return self._get_V()

  @V.setter
  def V(self, V):
    return self._set_V(V)

  def _get_V(self):
    if not self.in_V_rep and self.in_H_rep:
      self.determine_V_rep()
    return self._V

  def _set_V(self, V):
    # Support for singleton vertex definitions
    V = np.atleast_2d(np.asarray(V, dtype=float))

    if len(V.shape) > 2:
      raise ValueError('Vertices must be specified by a vector or a matrix '
        '(array of vectors)')

    nV, n = V.shape

    self._V = V
    self.n = n
    if nV:  # in_V_rep if V is not an empty array
      self.in_V_rep = True

  @property
  def nV(self):
    return self.V.shape[0]  # determines V-rep if not determined

  @property
  def centroid(self):
    return np.sum(self.V, axis=0) / self.nV

  @property
  def dim(self):
    if self._dim is None:
      self.determine_dim()
    return self._dim

  @property
  def is_full_dimensional(self):
    return self.dim == self.n

  def bbox(self):
    ''' Get the bounding box (minimum and maximum values) for the polytope '''
    if self.in_V_rep:
        bbox_min = np.min(self.V, axis=0)
        bbox_max = np.max(self.V, axis=0)
    else:
        # Need to use linear programming to determine the minima and maxima
        bbox_min = np.empty(self.n)
        bbox_max = np.empty(self.n)

        A = np.eye(self.n)

        for i, a in enumerate(A):
          # Maximum value for bounding box along axis
          res = linprog(-a, self.A, self.b.T, bounds=(-np.inf, np.inf))

          if res.success:
            bbox_max[i] = -res.fun
          else:
            raise RuntimeError('Could not determine bounding box. Optimization '
              'failed with the following message: \'{}\''.format(res.message))

          # Minimum value for bounding box along axis
          res = linprog(a, self.A, self.b, bounds=(-np.inf, np.inf))

          if res.success:
            bbox_min[i] = res.fun
          else:
            raise RuntimeError('Could not determine bounding box. Optimization '
              'failed with the following message: \'{}\''.format(res.message))

    return (bbox_min, bbox_max)

  def V_sorted(self):
    # Sort vertices (increasing angle: the point (x1, x2) = (1, 0) has angle 0).
    # np.arctan2(y, x) returns angles in the range [-pi, pi], so vertices are
    # sorted clockwise from 9:00 (angle pi). Note that the first argument is y.
    # Mainly for plotting and not implemented for n != 2.
    if self.n == 1:  # nothing to sort in 1-D
      if self.nV > 2:  # remove redundant vertices if there are more than two
        self.minimize_V_rep()
      return self.V
    if self.n != 2:
      raise NotImplementedError('V_sorted() not implemented for n != 2')
    c = self.centroid
    order = np.argsort(np.arctan2(self.V[:, 1] - c[1], self.V[:, 0] - c[0]))
    return self.V[order, :]

  def __repr__(self):  # TODO: does not print nicely for list of Polytopes
    r = [type(self).__name__]
    if not self:  # bool(self) = self.in_V_rep or self.in_H_rep
      r += [' (empty)']
    r += [f' in R^{self.n}']
    if self.in_H_rep:
      ineq_spl = 'inequalities' if self.A.shape[0] != 1 else 'inequality'
      r += [f'\n\tIn H-rep: {self.A.shape[0]} {ineq_spl}']
    if self.in_V_rep:
      vert_spl = 'vertices' if self.V.shape[0] != 1 else 'vertex'
      r += [f'\n\tIn V-rep: {self.nV} {vert_spl}']
    return ''.join(r)

  def __str__(self):
    return type(self).__name__ + f' in R^{self.n}'

  def __bool__(self):  # return True if the polytope is not empty
    return not self.is_empty()

  def __and__(self, other):  # return the intersection of self and other
    if isinstance(other, Polytope):
      # The intersection of an (empty) set with an empty set is an empty set, so
      # in this case, return an empty polytope of with dimension set to the
      # smaller of self.n and other.n (arbitrary choice).
      if not (self and other):
        return Polytope(n=min(self.n, other.n))
      else:
        return intersection(self, other)
    else:
      raise NotImplementedError('Intersection implemented for two polytopes '
                                'only')

  def __neg__(self):
    neg_P = Polytope()
    if self.in_V_rep:
      neg_P.V = -self.V
    if self.in_H_rep:
      neg_P._set_Ab(-self.A, self.b)
    return neg_P

  def __eq__(self, other):
    if isinstance(other, Polytope):
      return self.contains(other) and other.contains(self)
    else:
      raise ValueError('Equivalence of polytope and non-polytope not defined')

  def __le__(self, other):  # P <= other, other >= P
    if isinstance(other, Polytope):
      return other.contains(self)
    else:
      raise ValueError('The truth value of a polytope being a subset of a'
                       'non-polytope is not defined')

  def __ge__(self, other):  # P >= other, other <= P
    return self.contains(other)

  def __add__(self, other):
    if isinstance(other, Polytope):
      # The Minkowski sum of one non-empty and one empty polytope is the
      # non-empty polytope, so if one polytope is empty, return self (which can
      # be the empty one, possibly of different dimension than other):
      if not (self and other):
        return self._copy() if self else other._copy()
      else:  # compute the Minkowski sum when neither is empty
        return minkowski_sum(self, other)
    else:
      # TODO: handle zero vectors in a similar way (although now minimal overh.)
      return P_plus_p(self, other)

  def __radd__(self, other):
    return P_plus_p(self, other)

  def __sub__(self, other):
    if isinstance(other, Polytope):
      # The Pontryagin difference of one non-empty and one empty polytope is
      # the non-empty polytope, so if one polytope is empty, return self (which
      #  can be the empty one, possibly of different dimension than other):
      if not (self and other):
        return self._copy() if self else other._copy()
      else:  # compute the Pontryagin difference when neither is empty
        return pontryagin_difference(self, other)
    else:
      # TODO: handle zero vectors in a similar way (although now minimal overh.)
      return P_plus_p(self, other, subtract_p=True)

  # def __rsub__(self, other):  # p - P -- raise TypeError (not well defined)

  def __mul__(self, other):
    # self * other: scaling if other is scalar, inverse linear map if other is a
    # matrix (inverse=True has no effect on scaling).
    return self.multiply(other, inverse=True)

  def __rmul__(self, other):
    # other * self: scaling if other is scalar, linear map if other is a matrix.
    return self.multiply(other)

  def _copy(self):  # quick and dirty copy method
    # TODO: make proper ones (shallow and deep)
    P_copy = Polytope()
    if self.in_H_rep:
      P_copy._set_Ab(self.A, self.b)  # TODO: redesign this
    if self.in_V_rep:
      P_copy.V = self.V
    return P_copy

  def multiply(self, other, inverse=False):
    # scale: s * P or P * s with s a scalar and P a polytope
    # linear map: M * P with M a matrix
    # inverse linear map: P * M with M a matrix
    if isinstance(other, Polytope):
      raise NotImplementedError('Product of two polytopes not implemented')
    # TODO: now assuming a numeric type that can be squeezed -- fix
    # other can be a scalar (ndim=0), a vector (ndim=1), or a matrix (ndim=2)
    factor = np.squeeze(other)
    if factor.ndim == 0:
      return scale(self, other)
    elif factor.ndim in [1, 2]:
      if inverse:
        raise NotImplementedError('Inverse linear map P * M not implemented')
      else:
        return linear_map(other, self)
    else:
      raise NotImplementedError('Multiplication with numeric type other than '
                                'scalar and matrix not implemented')

  def determine_H_rep(self):
    if not self.in_V_rep:
      raise ValueError('Cannot determine H representation: no V representation')
    if not self.is_full_dimensional or self.n < 2:  # TODO: fix this
      # Use cdd, which handles this case.
      # cdd needs a column of ones to the left of V (zeros indicate rays):
      V_cdd = cdd.Matrix(np.hstack((np.ones((self.nV, 1)), self.V)),
                         number_type='float')
      V_cdd.rep_type = cdd.RepType.GENERATOR  # specifies that this is V-rep
      P_cdd = cdd.Polyhedron(V_cdd)
      H_cdd = P_cdd.get_inequalities()
      H_bmA = np.array(H_cdd)  # cdd's format of H is [b, -A]
      # Find the indices of inequalities Ax <= b (i_ineq_rows) and equations
      # Ax = b (i_eq_rows). lin_set is a a frozenset with indices of
      # equalities/equations Ax = b.
      i_ineq_rows = list(set(range(H_cdd.col_size)) - H_cdd.lin_set)
      i_eq_rows = list(H_cdd.lin_set)
      A_ineq = H_bmA[i_ineq_rows, 1:]
      b_ineq = H_bmA[[i_ineq_rows], [0]].T
      A_eq = H_bmA[i_eq_rows, 1:]
      b_eq = H_bmA[[i_eq_rows], [0]].T
      A = np.vstack((A_ineq, A_eq, -A_eq))
      b = np.vstack((b_ineq, b_eq, -b_eq))
      self._set_Ab(A, b)
    else: # use Qhull
      H = ConvexHull(self.V).equations  # Qhull's format of H is [A, -b]
      A, b_negative = np.split(H, [-1], axis=1)
      self._set_Ab(A, -b_negative)

  def determine_V_rep(self):  # also determines rays R (not implemented)
    # Vertex enumeration from halfspace representation using cdd.
    # TODO: shift the polytope to the center? (centroid? Chebyshev center?)
    # cdd uses the halfspace representation [b, -A] | b - Ax >= 0  ==>  Ax <= b.
    if not self.in_H_rep:
      raise ValueError('Cannot determine V representation: no H representation')
    b_mA = np.hstack((self.b, -self.A))  # [b, -A]
    H = cdd.Matrix(b_mA, number_type='float')
    H.rep_type = cdd.RepType.INEQUALITY  # specifies that this is H-rep
    H_P = cdd.Polyhedron(H)
    # From the get_generators() documentation: For a polyhedron described as
    #   P = conv(v_1, ..., v_n) + nonneg(r_1, ..., r_s),
    # the V-representation matrix is [t V] where t is the column vector with
    # n ones followed by s zeroes, and V is the stacked matrix of n vertex
    # row vectors on top of s ray row vectors.
    P_tV = H_P.get_generators()  # type(P_tV):  <class 'cdd.Matrix'>
    tV = np.array(P_tV[:])
    if tV.any():  # tV == [] if the Polytope is empty
      V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
      R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
      V = tV[V_rows, 1:]  # array of vertices (one per row)
      R = tV[R_rows, 1:]  # and of rays
      if R_rows.any():
        raise NotImplementedError('Support for rays not implemented')
    else:
      V = np.empty((0, self.n))
    self._set_V(V)

  def minimize_H_rep(self):
    # Minimize the number of halfspaces used to represent the polytope P by
    # removing redundant inequalities (rows) in A*x <= b. Determine the pair
    # (A, b) with the minimal number of rows and set self.A and self.b.
    redundant = redundant_inequalities(self.A, self.b)  # bool array
    self._set_Ab(self.A[~redundant], self.b[~redundant])  # TODO redesign _set

  def minimize_V_rep(self):
    # Minimize the number of vertices used to represent the polytope by removing
    # redundant points from the vertex list.
    # TODO: find and account for cases where this does not work
    # (1) Qhull does not support degenerate polytopes (not full dimensional),
    # such as lines in n = 2 amd planes in n = 3. ("QhullError: Raised when
    # Qhull encounters an error condition, such as geometrical degeneracy when
    # options to resolve are not enabled.")
    # (2) Qhull does not support 1-D polytopes ("ValueError: Need at least 2-D
    # data")
    if self.is_full_dimensional and self.n >= 2:
      # Indices of the unique vertices forming the convex hull:
      i_V_minimal = ConvexHull(self.V).vertices
      self.V = self.V[i_V_minimal, :]
    else:  # TODO: Use cdd for both cases?
      # cdd handles this case
      V_cdd = cdd.Matrix(np.hstack((np.ones((self.nV, 1)), self.V)),
                         number_type='float')
      V_cdd.rep_type = cdd.RepType.GENERATOR  # specify that this is V-rep
      V_cdd.canonicalize()  # removes redundant vertices
      # TODO: the lines below are copied from determine_V_rep() -- refactor
      tV = np.array(V_cdd)
      if tV.any():  # tV == [] if the Polytope is empty
        V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
        R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
        V = tV[V_rows, 1:]  # array of vertices (one per row)
        R = tV[R_rows, 1:]  # and of rays
        if R_rows.any():
          raise NotImplementedError('Support for rays not implemented')
      else:
        V = np.empty((0, self.n))
      self._set_V(V)

  def determine_dim(self):
    # Determine the dimensionality of the polytope using the vertices.
    # Subtract one vertex from the vertices so that the hyperplane that contains
    # the shifted vertices passes through the origin (it is then a subspace):
    V_subspace = self.V - self.V[0]  # subtract the first vertex
    # The number of linearly independent vertices is the dimension of that
    # subspace, and therefore the dimension of the polytope:
    self._dim = np.linalg.matrix_rank(V_subspace)

  def support(self, eta):
    # The support function of the polytope P, evaluated at (or in the direction)
    # eta in R^n:
    #   h_P(eta) = sup eta' * u   s.t.   u in P
    # See Eq. (1.11) in Kolmanovsky & Gilbert (1998), "Theory and computation of
    # disturbance  invariant sets for discrete-time linear systems."
    # Mathematical Problems in Engineering, 4(4), 317–367.
    result = solve_lp(-eta, A_ub=self.A, b_ub=self.b, bounds=(-np.inf, np.inf))
    h_P_eta = -result.fun  # the support is the optimal objective-function value
    return h_P_eta, result

  def project(self, dimensions):
    # Planar projection onto the plane specified in dimensions. E.g., to project
    # a polytope in 3-D onto the x1-x2 plane, dimensions should be (0, 1),
    # [0, 1], range(2), np.array([0, 1]). np.arange(2), or similar.
    dimensions = np.atleast_1d(np.squeeze(dimensions))
    # TODO: implement halfspace version
    if dimensions.ndim != 1:
      raise ValueError('dimensions parameter not in allowed form')
    if not self.in_V_rep:
      self.determine_V_rep()
    return Polytope(self.V[:, dimensions])

  def contains(self, Q):
    # If Q is a polytope: check whether Q is a subset of P (self). The test
    # is testing whether all vertices v of Q satisfy P.A * v <= P.b,
    # requiring the V-rep of Q and the H-rep of P. If both polytopes are in
    # H-rep, it is also straight forward to use the support function of Q for
    # the subset test. Using the V-rep of P (self) is a little more
    # complicated.
    # If Q is one or more points: check whether they are contained within P.
    if isinstance(Q, Polytope):  # P and Q are both polytopes
      if Q.n == self.n:  # TODO: test for more edge cases
        return all(self.contains(Q.V.T))  # can be extremely inefficient
      else:
        raise ValueError('Subset test of polytopes of different dimension not'
                         'allowed')
    else:
      # If Q contains m > 1 points: Q needs to have shape (P.n, m). That is,
      # each of the points is a column.
      # If Q contains a single point, it can have any shape and be of any type
      # that can be converted to an ndarray as below.
      # Try to convert Q to an ndarray (raises ValueError if Q is not numeric):
      x = np.atleast_2d(np.squeeze(Q).astype(float))  # x: one or more points
      if x.shape[0] == 1:  # make x a column if it is single point (here a row)
        x = x.T
      elif x.shape[0] != self.n:
        # Could transpose here if x.shape[1] == self.n, but better to specify
        # that points must be columns:
        raise ValueError(f'Testing if a polytope in R^{self.n} contains points'
                         f'in R^{x.shape[0]} is not allowed: points must be '
                         f'columns (transpose the input?)')
      
      if self.in_H_rep:
        # Check whether P.A @ x <= P.b (with a tolerance) for (all) x. Iff that
        # is the case, the points x are all contained in P. To test this, check
        # for which x P.A @ x - P.b <= 0, with a tolerance, and return whether it
        # holds for all x.
        Ax_m_b = self.A @ x - self.b
        # Test whether every point satisfies every inequality. The following gives
        # an (P.n, x.shape[1] bool array:
        x_contained = np.logical_or(Ax_m_b < 0, np.isclose(Ax_m_b, 0))
        # Return an array of which points that satisfy every inequality, and are
        # thus contained in P:
        return np.all(x_contained, axis=0)


  def is_empty(self):
    # First check if A, b, and V are all empty
    if self.A.shape[0] == 0 and self.b.shape[0] == 0 and self.V.shape[0] == 0:
      return True

    # If V representation is non-empty then polyhedron is trivially nonempty
    if self.V.shape[1] == self.n:
      return False

    # If H-representation that has not been reduced, it is possible to create
    # empty polyhedra, e.g. from intersections of disjoint sets.
    # Can find out if it is empty through simple LP.
    res = linprog(np.zeros(size.n), self.A, self.b)

    if res.status == 0:
      return False
    else:
      # TODO: Catch edge-cases for which optimization solver doesn't "terminate
      # successfully" but problem is not actually infeasible. Since the 
      # objective function is a constant, this should avoid unboundedness
      # res.status == 3.
      return True

    

  def plot(self, ax=None, **kwargs):
    # Plot Polytope. Add separate patches for the fill and the edge, so that
    # the fill is below the gridlines (at zorder 0.4) and the edge edge is
    # above (at zorder 2, same as regular plot). Gridlines are at zorder 1.5
    # by default and at  (1 is default), which is below gridlines, which are
    # at zorder 1.5 by default 0.5 if setting ax.set_axisbelow(True),
    # so plotting the fill at 0.4 ensures the fill is always below the grid.
    if not ax:
      ax = plt.gca()
    h_patch = []  # handle, return as tuple
    V_sorted = self.V_sorted()
    # 1-D polytopes are lines (segments) and need a second dimension for
    # plotting. For now assumes a 1-D polytope has vertices along the first
    # coordinate.
    if self.n == 1:  # add zero-coordinates for second dimension (lift to n = 2)
      V_sorted = np.hstack((V_sorted, np.zeros((self.nV, 1))))
    # Check for edgecolor. Default is (0, 0, 0, 0), with the fourth 0 being
    # alpha (0 is fully transparent). Passingedgecolor='r', e.g., later
    # translates to (1.0, 0.0, 0.0, 1).
    edgecolor_default = (0, 0, 0, 0)
    edgecolor = kwargs.pop('edgecolor', edgecolor_default)
    # Handle specific cases: edge transparent or set explicitly to None
    if ((type(edgecolor) is tuple and len(edgecolor) == 4 and edgecolor[3] == 0)
        or edgecolor is None):
      edgecolor = edgecolor_default
    # Added keyword for edge transparency: edgealpha (default: 1.0)
    edgealpha_default = 1.0
    edgealpha = kwargs.pop('edgealpha', edgealpha_default)
    # Check for fill and facecolor. fill is default True. The Polygon
    # documentation lists facecolor=None as valid but it results in blue
    # filling (preserving this behavior).
    fill = kwargs.pop('fill', True)
    facecolor = kwargs.pop('facecolor', None)
    # test for any non-empty string and rbg tuple, and handle black as rbg
    if any(edgecolor) or edgecolor == (0, 0, 0):
      # Plot edge:
      temp_dict = {**kwargs,
                   **{'fill': False, 'edgecolor': edgecolor, 'zorder': 2,
                      'alpha': edgealpha}}
      h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
    if fill or facecolor:
      # Plot fill:
      temp_dict = {**kwargs,
                   **{'edgecolor': None, 'facecolor': facecolor, 'zorder': 0.4}}
      h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
    return tuple(h_patch)  # handle(s) to the patch(es)

  def plot_basic(self, ax, **kwargs):
    h_patch = ax.add_patch(Polygon(self.V_sorted(), **kwargs))
    return h_patch  # handle to the patch


def P_plus_p(P, point, subtract_p=False):
  # Polytope + point: The sum of a polytope in R^n and an n-vector -- P + p
  # If subtract_p == True, compute P - p instead. This implementation allows
  # writing Polytope(...) - (1.0, 0.5) or similar by overloading __sub__ with
  # this function.
  # TODO: rename -- this is also a Minkowski sum
  p = np.array(np.squeeze(point), dtype=float)[:, np.newaxis]
  if p.size != P.n or p.shape[1] != 1 or p.ndim != 2:  # ensure p is n x 1
    raise ValueError(f'The point must be a vector in R^{P.n}')
  if subtract_p: p = -p  # add -p if 'sub'
  P_shifted = Polytope()
  # V-rep: The sum is all vertices of P shifted by p.
  # Not necessary to tile/repeat since p.T broadcasts, but could do something
  # like np.tile(q.T, (P.nV, 1)) or np.repeat(p.T, P.nV, axis=0).
  if P.in_V_rep:
    P_shifted.V = P.V + p.T
  # H-rep: A * x <= b shifted by p means that any point q = x + p in the shifted
  # polytope satisfies A * q <= b + A * p, since
  #   A * x <= b  ==>  A * (q - p) <= b  ==>  A * q <= b + A * p
  # That is, the shifted polytope has H-rep (A, b + A * p).
  if P.in_H_rep:
    P_shifted._set_Ab(P.A, P.b + P.A @ p)  # TODO: redesign this
  return P_shifted


def minkowski_sum(P, Q, reduce=True):
  # Minkowski sum of two convex polytopes P and Q:
  # P + Q = {p + q in R^n : p in P, q in Q}.
  # In vertex representation, this is the convex hull of the pairwise sum of all
  # combinations of points in P and Q.
  if P.n != Q.n:
    raise ValueError(f'Minkowski sum of polytopes of different dimensions '
                     f'({P.n} and {Q.n}) not allowed')
  # TODO: add more tests on P and Q (both non-empty? ...)
  # TODO: find minimal V-reps? or should be up to caller?
  # Vertices of the Minkowski sum:
  msum_V = np.full((P.nV * Q.nV, P.n), np.nan, dtype=float)
  for i_q, q in enumerate(Q.V):  # TODO: loop over the smallest vertex set?
    msum_V[i_q * P.nV: (i_q + 1) * P.nV, :] = P.V + q
  P_plus_Q = Polytope(msum_V)  # TODO: make this more compact with chaining?

  if reduce:
    P_plus_Q.minimize_V_rep()
    
  return P_plus_Q


def pontryagin_difference(P, Q, reduce=True):
  # Pontryagin difference for two convex polytopes P and Q:
  # P - Q = {x in R^n : x + q in P, for all q in Q}
  # In halfspace representation, this is [P.A, P.b - Q.support(P.A)], with
  # Q.support(P.A) a matrix in which row i is the support of Q at row i of P.A.
  if P.n != Q.n:
    raise ValueError(f'Pontryagin difference of polytopes of different '
                     f'dimensions ({P.n} and {Q.n}) not allowed')
  m = P.A.shape[0]
  pdiff_b = np.full(m, np.nan)  # b vector in the Pontryagin difference P - Q
  # For each inequality i in P: subtract the support of Q in the direction P.A_i
  for ineq in range(m):
    pdiff_b[ineq] = P.b[ineq] - Q.support(P.A[ineq])[0]  # TODO: error check [1]
  
  if reduce:
    # Determine which inequalities are redundant, if any:
    redundant = redundant_inequalities(P.A, pdiff_b)
    pdiff = Polytope(P.A[~redundant], pdiff_b[~redundant])  # minimal H-rep
  else:
    pdiff = Polytope(P.A, pdiff_b)

  return pdiff


def scale(P, s):
  # TODO: handle s == 0 and s == 1 separately
  P_scaled = Polytope()
  if P.in_H_rep:
    P_scaled._set_Ab(P.A, P.b * s)  # TODO: redesign this
  if P.in_V_rep:
    P_scaled.V = P.V * s
  return P_scaled


def linear_map(M, P):
  # Compute the linear map M * P through the vertex representation of V. If P
  # does not have a vertex representation, determine the vertex representation
  # first. In this case, the H-representation of M * P is NOT determined before
  # returning M * P.
  n = M.shape[1]
  if P.n != n:
    raise ValueError('Dimension of M and P do not agree in linear map M * P')
  # TODO: implement linear map in H-rep for invertible M
  # TODO: M_P = Polytope(P.V @ M.T), if P.in_H_rep: M_P.determine_H_rep()?
  return Polytope(P.V @ M.T)


def intersection(P, Q, reduce=True):
  # Set intersection of the polytopes P and Q. P_i_Q: P intersection Q
  # TODO: improve handling of empty intersections
  # Combine the H-representation of both polytopes:
  P_i_Q_A = np.vstack((P.A, Q.A))
  P_i_Q_b = np.vstack((P.b, Q.b))

  if reduce:
    # Determine which inequalities are redundant, if any:
    redundant = redundant_inequalities(P_i_Q_A, P_i_Q_b)
    # Create the intersection P_i_Q from the inequalities that are not redundant:
    P_i_Q = Polytope(P_i_Q_A[~redundant], P_i_Q_b[~redundant])
  else:
    P_i_Q = Polytope(P_i_Q_A, P_i_Q_b)

  return P_i_Q


def redundant_inequalities(A, b):
  # Identify redundant inequalities (rows) in A*x <= b. Determine the pair
  # (A, b) with the minimal number of rows and return which inequalities that
  # are removed to reduce the input (A, b) to the minimal pair. See Komei
  # Fukuda's polytope FAQ for an explanation of the redundancy test:
  # http://www.cs.mcgill.ca/~fukuda/download/paper/polyfaq.pdf Returns: -
  # removed: bool array with True at every redundant constraint
  m, n = A.shape
  # Add I to b so that column j of the resulting matrix is b + e_j (unit vector)
  b_plus_1 = np.array(np.squeeze(b), dtype=float)[:, np.newaxis] + np.eye(m)
  x_bounds = (-np.inf, np.inf)  # 0 <= x is the default in SciPy's linprog
  removed = np.full(m, False)  # use to index A and b as LP constraints
  # TODO: first remove inequalities with all 0s in row i of A or infinity in b_i
  for ineq in range(m):
    lp_result = solve_lp(-A[ineq, :], A_ub=A[~removed, :],
                         b_ub=b_plus_1[~removed, ineq], bounds=x_bounds)
    if not lp_result.success:
      print(f'Tested inequality {ineq}: {lp_result.message}')
    # The constraint a_i' * x <= b is redundant iff a_i' * x <= b_i at the
    # solution (the maximum of a_i * x, which is -lp_result.fun). Test that
    # -lp_result.fun <= b_i with a tolerance (use numpy's standard atol, rtol):
    if -lp_result.fun <= b[ineq] or np.isclose(-lp_result.fun, b[ineq]):
      removed[ineq] = True
  return removed


def solve_lp(c, solver='linprog', *args, **kwargs):
  # Wrapper for various LP solvers (currently only scipy.optimize.linprog).
  if solver.lower() == 'linprog':
    kwargs['method'] = 'revised simplex'
    result = linprog(c, **kwargs)
    if not result.success:
      print({result.message})
  else:
    raise NotImplementedError(f'Support for solver {solver} not implemented')
  return result

def poly_slice(P, dims, values, keep_dims=False):
  ''' Polytope slicing

  Based off the implementation in Multi-Parametric Toolbox (MPT3)
  '''
  if P.is_empty():
    return Polytope(n=P.dim)
  
  if any(dims >= P.dim):
    raise ValueError('Value in dimension array exceeds total Polytope '
      'dimension')

  dims = np.array(dims, dtype=float)
  values = np.array(values, dtype=float)

  if dims.shape[0] != values.shape[0]:
    raise ValueError('Dimension array and value array must be 1-d vectors '
      'of equivalent length')

  if len(dims.shape) > 1 or len(values.shape) > 1:
    raise ValueError('Dimension array and value array must be 1-d vectors')

  # Need H-representation in order to perform the slice
  # The getter for A will determin the H-representation if it does not already
  # exist.
  if keep_dims:
    # Return a polytope of the same "dimension" of the original.
    # In this context, dimension is different than span as defined in the
    # original class
    Z = np.zeros((dims.shape[0], P.dim), dtype=float)
    for i in range(Z.shape[0]):
      Z[i][dims[i]] = 1.0

    return Polytope(np.vstack((P.A, P.Z, -P.Z)), 
      np.concatenate(P.b, values, -values))

  else:
    Z = np.empty((0, P.dim-dims.shape[0]))
    c = np.empty(P.dim-dims.shape[0])

    # Array with zeros except for the values specified in dims
    x = np.zeros(P.dim)
    for i in range(dims):
      x[dims[i]] = values[i]

    AT = np.transpose(P.A)
    for i in range(P.dim):
      c = np.concatenate((c, P.b[i] - P.A[i] @ x))
      
      if not i in dims:
        # Because of slicing issues, it's easier to stack the transposed matrix
        # and then re-transpose at the end
        Z = np.vstack((Z, AT[i]))

    # Transpose the matrix back
    Z = np.transpose(Z)

    return Polytope(Z, c)

class NdBox(Polytope):
  ''' NdBox

  An NdBox (n-dimensional hypercube) is a specialized type of polytope that
  is defined by a lower-bound and and upper-bound. These are commonly used
  sets to define basic rectangular bounds in spaces. Because of their
  definitions, some set operations can be performed exactly and quickly.

  '''
  def __init__(self, lb, ub):
    ''' NdBox Constructor

    Args:
      lb - Lower bound, array-like object
      ub - Upper bound, array-like object

    '''
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if len(lb.shape) > 1 or len(ub.shape) > 1:
      raise ValueError('Lower and upper bounds must be arrays')

    self._lb = lb
    self._ub = ub

    super().__init__(lb=lb, ub=ub)

  @property
  def lb(self):
    ''' Lower bound of NdBox '''
    return self._lb

  @property
  def ub(self):
    ''' Upper bound of NdBox '''
    return self._ub

  def __and__(self, B):
    ''' Intersection '''
    if isinstance(B, NdBox):
      lb = np.max((self.lb, B.lb))
      ub = np.min((self.ub, B.ub))

      return NdBox(lb, ub)
    else:
      return super().__and__(B)

  def __radd__(self, B):
    return self.__add__(B)

  def __add__(self, B):
    if isinstance(B, (list, tuple)):
      B = np.asarray(B, dtype=float)

    if isinstance(B, np.ndarray):
      if len(B.shape) == 1:
        return NdBox(self.lb+B, self.ub+B)
      else:
        raise ValueError('Cannot add NdBox and np.ndarray of shape {}'.format(
          B.shape))
    elif isinstance(B, NdBox):
      return NdBox(self.lb+B.lb, self.ub+B.ub)
    else:
      return super().__add__(B)

  def __minus__(self, B):
    ''' Pontryagin difference '''
    if isinstance(B, (list, tuple)):
      B = np.asarray(B, dtype=float)

    if isinstance(B, np.ndarray):
      return self.__add__(-B)
    elif isinstance(B, NdBox):
      xcs = (self.ub + self.lb) / 2
      xcB = (B.ub + B.lb) / 2

      lb = self.lb + (B.ub - B.lb) + (xcs - xcB)
      ub = self.ub + (B.lb - B.ub) + (xcs - xcB)

      if np.any(ub < lb):
        return Polytope(n=self.n)
      else:
        return NdBox(lb, ub)

    else:
      return super().__minus__(B)

  def __mul__(self, M):
    return NotImplemented
  
  def __rmul__(self, M):
    return self.matmul(M)

  def matmul(self, M, ndbox_approx=None):
    ''' Matrix multiply

    This function implements M * B, where M is a matrix and B is an NdBox. 
    Typically this multiplication will result in the set no longer being an
    NdBox, but the results approximation can be over and underapproximated with
    an NdBox very quickly. This is useful for high-dimensional polytope 
    computations where vertex-facet enumeration becomes limiting.

    Args:
      M - Multiplying Matrix

    Kwargs:
      ndbox_approx: None - The type of approximation method to perform; can be
          either None, 'Over', or 'Under'

    '''
    if isinstance(M, (int, float)):
      if M < 0:
        return NdBox(M * self.ub, M * self.lb)
      else:
        return NdBox(M * self.lb, M * self.ub)

    elif isinstance(M, np.ndarray):
      if len(M.shape) != 2:
        raise ValueError('Cannot perform multiplication M * B for M with '
          'len(M.shape) !=2')

      if ndbox_approx == None:
        return super().multiply(M)
      else:
        if not isinstance(ndbox_approx, str):
          raise TypeError('Invalid type for ndbox_approx. Received `{}` but '
            'expecting String or None'.format(type(ndbox_approx)))

        if ndbox_approx.lower() == 'over':
          # create a box over-approximation
          ub = [self.support(e) for e in M @ np.eye(self.n).T]
          lb = [-self.support(e) for e in -M @ np.eye(self.n).T]

          return NdBox(lb, ub)
        elif ndbox_approx.lower() == 'under':
          raise NotImplementedError('Working on it...')
        else:
          raise ValueError('Invalid option for ndbox_approx. Received `{}` '
            'but expected None, `Over`, or `Under`'.format(ndbox_approx))

    else:
      return NotImplemented

  def support(self, l):
    ''' Support function value of NdBox

    Args:
      l - Direction vector

    '''
    l = np.asarray(l, dtype=float)

    if len(l.shape) > 1:
      raise ValueError('Support direction must be an array')

    if l.shape[0] != self.n:
      raise ValueError('Support direction not of the same dimension as the '
        'NdBox: {} != {}'.format(l.shape[0], self.n))

    v = np.empty(self.n)
    for i in range(self.n):
      v[i] = self.ub[i] if l[i] >=0 else self.lb[i]

    return l @ v

  def contains(self, x):
    ''' Check if point/set is contained in Ndbox

    Args:
      x - Array or set

    '''
    if isinstance(x, (list, tuple)):
      x = np.asarray(x, dtype=float)

    if isinstance(x, np.ndarray):
      if len(x.shape) > 1:
        if len(x.shape) > 2:
          raise ValueError('Cannot check containment for np.ndarray of '
            'shape {}'.format(x.shape))
        
        return np.all(x <= self.ub, axis=1) * np.all(x >= self.lb, axis=1)

      else:
        return np.all(x <= self.ub) * np.all(x >= self.lb)

    else:
      return super().contains(x)

  def sample(self):
    ''' Uniform sample inside box '''
    return np.random.uniform(self.lb, self.ub)

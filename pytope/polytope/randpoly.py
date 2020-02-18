class RandomPolytope(Polytope):
  ''' RandomPolytope class

  A random polytope is a special class of polytope in which the vertex / facet
  enumerations are performed through random generation. These types of 
  generation methods converge to the true polytope as n increases to infinity,
  see, for example: Wilfrid S. Kendall and Ilya Molchanov, "New Perspectives in 
  Stochastic Geometry," Oxford University Press, 2010.

  Random polytopes can quickly approximate polytopes of high dimension, where
  traditional vertex / facet enumeration methods---e.g. CDD, LRS, 
  polymake---become computationally infeasible.
  '''
  def __init__(self, *args, **kwargs):
    ''' RandomPolytope Constructor

    For 

    Additional Keyword arguments:
      - nH
      - nV
    '''
    super().__init__(*args, **kwargs)

    self._nV = kwargs['nV'] if 'nV' in kwargs else None
    self._nH = kwargs['nH'] if 'nH' in kwargs else None

  @property
  def nV(self):
    ''' Number of vertices for randome vertex generation '''
    return self._nV

  @nV.setter
  def nV(self, val):
    if val - math.floor(val) > 0:
      raise ValueError('Number of vertices argument must be integer-like')

    self._nV = val

  @property
  def nH(self):
    ''' Number of half-spaces used for random facet generation '''
    return self._nH

  @nH.setter
  def nH(self, val):
    if val - math.floor(val) > 0:
      raise ValueError('Number of half spaces argument must be integer-like')

    self._nH = val

  def determine_V_rep(self, verbose=0):
    # Vertex enumeration using probabilistically convergent random sampling 
    # method. Useful for high-dimensional polytopes in which full enumeration
    # with cdd becomes intractable.

    if not self.in_H_rep:
      raise ValueError('Cannot determine random V representation: no H '
                       'representation')

    # Random vertex algorithm
    #   1) Find the chebychev center of the polytope, see Section 4.3.1 Boyd
    #      and Vendenberghe "Convex Optimization"
    #   2) Generate new vertices
    #     2a) Randomly generate multivariate normal
    #     2b) Find scaling that satisfies polytope containment
    # 

    # Chebychev center LP
    # TODO: Integrate with solve_lp
    if verbose > 0:
      print('Computing Chebychev center... ', end='')
      tick = time.process_time()

    res = linprog(np.concatenate((np.zeros(self.n), [-1])), 
      np.concatenate((self.A, [[np.linalg.norm(a)] for a in self.A]), axis=1), 
      self.b)

    if verbose > 0:
      print('{} seconds'.format(time.process_time() - tick))

    xc = res['x'][:-1]

    # Setup new polytope that contains the origin
    P = self - xc

    # Solve for vertices
    # TODO: Look at parallelization
    # Initial matrix fill
    if verbose > 0:
      print('Computing random sample vertex form... ', end='')
      tick = time.process_time()

    V = np.zeros((self.nV, self.n))
    for i in range(V.shape[0]):
      # Random vector generation
      v = np.random.multivariate_normal(np.zeros(self.n), np.eye(self.n))

      # Scaling for containment
      theta = np.min(np.abs(P.b / (P.A @ v)))
      v *= theta

      # Shift vertex back by Chebychev center
      V[i] = v

    if verbose > 0:
      print('{} seconds'.format(time.process_time() - tick))

    P.V = V
    P = P + xc

    return P

  def determine_H_rep(self):
    # Facet enumeration using probabilistically convergent random sampling 
    # method. Useful for high-dimensional polytopes in which full enumeration
    # with cdd becomes intractable.

    if not self.in_H_rep:
      raise RuntimeError('Cannot determine random H representation: no V '
                       'representation')

    if self.nH is None:
      raise RuntimeError('Cannot determine random H representation: number '
                         'of facet, nH, not specified')

    if self.nH < 2 * self.n:
      raise RuntimeError('Cannot determine random H representation: minimum '
                         'number of facets must be twice the dimension of the '
                         'polytope')

    # Random facet algorithm
    #   1) Generate bounding box to ensure compactness of resulting set
    #   2) Generate random vector from multivariate normal, ai
    #   3) Find b such that ai^T * V \leq b for all vectors in V
    #   4) Repeat
    # 

    # Initialize H representation A and b
    A = np.zeros((self.nH, self.n))
    b = np.zeros(self.nH)

    # Bounding box
    A[:self.n, :] = np.eye(self.n)
    A[self.n+1:2*self.n, :] = -np.eye(self.n)
    b[:self.n] = self.V.max(axis=0)
    b[self.n+1:2*self.n] = -self.V.min(axis=0)

    for i in range(self.nH - 2 * self.n):
      # Generate random direction
      ai = np.random.multivariate_normal(np.zeros(self.n), np.eye(self.n))

      A[2*self.n+1+i, :] = ai
      b[2*self.n+1+i] = np.max(self.V @ ai)

    return Polytope(A, b, V=self.V)

  def minimize_V_rep(self):
    return NotImplementedError('Support not currently implemented')

  def minimize_H_rep(self):
    return NotImplementedError('Support not currently implemented')
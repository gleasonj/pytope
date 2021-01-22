import numpy as np

from ..polytope import Polytope
from .lazypolytopes import LazyVertexSet, LazyFacetSet

class SupportSet():
    def __add__(self, B):
        if isinstance(B, SupportSet):
            if B.dim != self.dim:
                raise ValueError('Dimension mismatch between added SupportSets')

            return MinkowskiAdditionSupportSet(self, B)
        elif isinstance(B, np.ndarray) and len(B.shape) == 1:
            if len(B) != self.dim:
                raise ValueError('Dimension mistmatch between SupportSet and '
                    'numpy array')

            return MinkowskiAdditionSupportSet(self, SingletonSupportSet(B))
        elif isinstance(B, LazyVertexSet):
            return MinkowskiAdditionSupportSet(self, LazyVertexSuportSet(B))
        else:
            return NotImplemented

    def __call__(self, l: np.ndarray):
    ''' Base method to obtain the support vector for the support set. This 
    method is typically not called and it is preferred for users to use the
    supvec function.

    The calling function assumes that the provided input is a 1-d numpy array
    with a magnitude (2-norm) of 1. The call functions are not typically written
    with robust input checks because it is assumed that users have used the
    supvec function, which has the appropriate input handling.
    '''
        raise NotImplementedError

def SingletonSupportSet(SupportSet):
    def __init__(self, x: np.ndarray):
        self._x = x

    @property
    def x(self):
        return self._x

    def __call__(self, l: np.ndarray):
        return self.x

def LazyVertexSupportSet(SupportSet):
    def __init__(self, S: LazyVertexSet):
        self._S = S

    def __call__(self, l: np.ndarray):
        self._S[np.argmax(l @ self._S)]

def MinkowskiAdditionSupportSet(SupportSet):
    def __init__(self, A: (SupportSet, LazyVertexSet, Polytope), 
        B: (SupportSet, LazyVertexSet, Polytope)):
    
        self._A = A
        self._B = B

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    def __call__(self, l: np.ndarray):
        return self.A(l) + self.B(l)

def MatMulSupportSet

class Ballp(SupportSet):
''' A Euclidean p-norm Ball Set

The Euclidean p-norm Ball set is defined as

    { x \in R^n | ||(x - c)||_p <= r }

INPUTS:
    p   Norm value
    center  Center point
    radius  Radius value
'''
    def __init__(self, p: int, center: np.ndarray, radius: float):
        assert len(center.shape) == 1, 'Center must be an array'
        self._p = p
        self._center = center
        self._radius = radius

    @property
    def center(self):
    ''' Center point of the Ballp set '''
        return self.center

    @property
    def radius(self):
    ''' Radius of the Ballp set '''
        return self._radius

    @property
    def dim(self):
    ''' Dimension of the Ballp set '''
        return len(self._center)

    def contains(self, x: np.ndarray):
    ''' Check is a point or collection of points is contained in the set. '''
        assert len(x.shape) < 3, \
            'Point(s) must be an array or list of arrays (1 or 2-d numpy ' \
            'arrays)'
        
        if len(x.shape) == 2:
            return np.array([self.contains(v) for v in x])
        else:
            if np.linalg.norm(x - self.center, ord=self.p) <= self.radius:
                return True
            else:
                return False

class Ball2(Ballp):
''' A Euclidean Ball set with p=2 '''
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(2, center, radius)

    def __call__(self, l: np.ndarray):
    ''' Return the support vector for a 1-d numpy array of unit magnitude. '''
        return self.center + self.radius * l    


def supvec(S, l: np.ndarray):
''' Obtain the support vector(s) for given set and direction(s).

While the various implementations of support sets have a call function, this is
the primary function that users should use to obtain support vectors as it 
properly handles different inputs, e.g. multiple direction vectors or
direction vectors that do not have unit magnitude.

The support vector is defined as:

        argmax  l^T @ x
        w.r.t.  x 
    subject to  x \in S

NOTE: While l and x are mathematically defined as vectors in n-dimensional 
      Euclidean space, because of how arrays are typically created in numpy,
      a vector is a 1-d numpy array and a collection of vectors is given
      by a 2-d numpy array where each row is a vector. Values are returned 
      by the same convention.

INPUTS:
    S   The given set for which to determine the support vector. The set must
        be amenable to determination of support vectors. Currently supported
        types are:
            SupportSet (with an appropriately defined call function)
            Polytope
            LazyVertexSet
            LazyFacetSet

    l   A 1-d numpy array of the given direction or 2-d numpy array for which
        each row represents a direction.

OUTPUTS:
    v   A 1-d numpy array of the support vector or 2-d numpy array for which
        each row represents a support vector.
'''
    assert len(l.shape) < 3, 'Direction(s) must be a 1-d numpy array or a ' \
        '2-d numpy array where each row is a direction vector.'

    # Recursive call to handle set of directions
    if len(l.shape) == 2:
        return np.array([supvec(S, v) for v in l])

    # 1-d direction vector
    if isinstance(S, SupportSet):
        if np.linalg.norm(l) == 0:
            raise ValueError('Cannot compute support vector for zero direction '
                'vector.')

        return S(l / np.linalg.norm(l))
    elif isinstance(S, Polytope):
        raise NotImplementedError('Working on it...')
    elif isinstance(S, LazyVertexSet):
        return S[np.argmax(l @ S.V.T)]
    elif isinstance(S, LazyFacetSet):
        raise NotImplementedError('Working on it...')
    else:
        raise NotImplementedError('Set type {} not supported.'.format(type(S)))

def supfcn(S, l: np.ndarray):
''' Obtain the support function value(s) for given set and direction(s).

A support function is defined as:

        max  l^T @ x
        w.r.t.  x 
    subject to  x \in S

Computing the support function is equivalent to calling l @ supvec(S, l).T.

INPUTS:
    S   The given set for which to determine the support vector. The set must
        be amenable to determination of support vectors. Currently supported
        types are:
            SupportSet (with an appropriately defined call function)
            Polytope
            LazyVertexSet
            LazyFacetSet

    l   A 1-d numpy array of the given direction or 2-d numpy array for which
        each row represents a direction.

OUTPUTS:
    v   A float or 1-d array of support function values.
'''
    return l @ supvec(S, l).T

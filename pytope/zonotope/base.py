import numpy as np

from pytope import EmptySet

class Zonotope():
    ''' Zonotope class
    
    A Zonotope is a special type of vertex polytope defined by a center, c, and 
    generator vectors, g_{i}.

        Z = { x \in \mathbb{R}^{n} : x = c + \lambda_{i} g_{i}, 
                \lambda_{i} \in [-1, 1] }

    '''
    # TODO: Implement Minkowski (Pontryagin) difference
    # TODO: Implement to_polytope

    __array_ufunc__ = None

    def __init__(self, c, g=None):
        ''' Zonotope class constructor '''

        c = np.asarray(c, dtype=float)
        if len(c.shape) > 1:
            raise ValueError('Zonotope center must be an array')

        self._c = c

        if not g is None:
            g = np.asarray(g, dtype=float)

            if len(g.shape) == 1:
                if g.shape[0] != c.shape[0]:
                    raise ValueError('Center and generator vectors are not '
                        'of the same dimension: {} != {}'.format(g.shape[0],
                        c.shape[0]))

                self._g = np.atleast_2d(g)
            elif len(g.shape) == 2:
                if g.shape[1] != c.shape[0]:
                    raise ValueError('Center and generator vectors are not '
                        'of the same dimension: {} != {}'.format(g.shape[1],
                        c.shape[0]))

                self._g = g
            else:
                raise ValueError('Cannot create zonotope with generator '
                    'matrix of shape {}'.format(g.shape))

        else:
            self._g = np.zeros((1, self.n))

    @property
    def c(self):
        ''' Zonotope center '''
        return self._c


    @property
    def g(self):
        return self._g

    @property
    def n(self):
        return self._c.shape[0]

    def __add__(self, B):
        if isinstance(B, (list, tuple)):
            B = np.asarray(B, dtype=float)

        if isinstance(B, Zonotope):
            # Minkowski summation of two zonotopes
            if self.n != B.n:
                raise RuntimeError('Cannot perform Minkowki Summation: Zonotopes '
                    'are not the same dimension')

            return Zonotope(self.c + B.c, np.vstack((self.g, B.g)))
            
        elif isinstance(B, np.ndarray):
            if len(B.shape) > 1:
                raise ValueError('Cannot add Zonotope and array of shape '
                    '{}'.format(B.shape))
                    
            return Zonotope(self.c+B, self.g)

        elif isinstance(B, EmptySet):
            return Zonotope(self.c, self.g)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, M):
        return NotImplemented
    
    def __rmul__(self, M):
        return self.matmul(M)

    def matmul(self, A):
        A = np.asarray(A, dtype=float)

        return Zonotope(A @ self.c, self.g @ A.T)

    def to_polytope(self):
        ''' Convert zonotope to vertex polytope '''
        raise NotImplementedError('Working on it...')

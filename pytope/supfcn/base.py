import numpy as np

import math

from pytope.polytope import Polytope

from scipy.optimize import linprog

class SupportFcn():
    ''' Support Function Class

    Support Functions are a method of describing convex sets. They are defined
    by 

        S(P; l) = sup {l^{T} x | x \in P }

    for some convex set P. They are used for the efficiency in representing
    Minkowski addition and linear mapping operations. See Boyd and Vandenberghe,
    "Convex Optimization".
    '''

    # TODO: Add documentation
    # TODO: Additional unit testing
    # TODO: Add support for Minkowski differences between Polytopes and Support
    #       functions
    # TODO: Implement support vectors

    # Turn off numpy ufunc
    __array_ufunc__ = None

    def __init__(self, n, callback=None):
        self._callback = callback
        self._n = n

    @property
    def n(self):
        return self._n

    def __call__(self, l):
        if self.is_empty:
            return np.array([], dtype=float)
        else:
            return self._callback(l)

    def __add__(self, b):
        if isinstance(b, SupportFcn):
            if self.n != b.n:
                raise ValueError('Cannot add support functions of different '
                    'dimensions: {} + {}'.format(self.n, b.n))

            if self.is_empty:
                if b.is_empty:
                    return SupportFcn(self.n)
                else:
                    return SupportFcn(b.n, callback=lambda l: b(l))
            else:
                if b.is_empty:
                    return SupportFcn(self.n, callback=lambda l: self(l))
                else:
                    return SupportFcn(self.n, callback=lambda l: self(l) + b(l))

        elif isinstance(b, np.ndarray):
            if len(b.shape) > 1:
                raise ValueError('Cannot add support function and a matrix')
            else:
                if b.shape[0] != self.n:
                    raise ValueError('Cannot perform addition: support '
                        'function and vector are not of compatible dimension: '
                        '{} + {}'.format(self.n, b.shape[0]))
                
            return SupportFcn(self.n, callback=lambda l: self(l) + (l @ b))

        elif isinstance(b, float):
            if self.n > 1:
                raise ValueError('Cannot perform addition: support function '
                    'dimension mismatch: {} + {}'.format(self.n, 1))

            return SupportFcn(self.n, callback=lambda l: self(l) + l * b)
        else:
            return NotImplemented

    def __rmul__(self, A):
        if self.is_empty:
            return SupportFcn(self.n)

        if isinstance(A, np.ndarray) or isinstance(A, float):
            if isinstance(A, float):
                return SupportFcn(self.n, callback=lambda l: self(A * l))
            else:
                if len(A.shape) != 2:
                    if len(A.shape) == 1 and A.shape[0] == 1:
                        # Passed a scalar in the form of a numpy array
                        return A[0] * self
                    else:
                        raise ValueError('Can only perform multiplication by '
                            'a scalar or a matrix')
                    
                else:
                    if self.n != A.shape[1]:
                        raise ValueError('Dimension mismatch between matrix '
                            'multiplier and dimension of the support function '
                            'space')

                    return SupportFcn(A.shape[0], 
                        callback=lambda l: self(A.T @ l))
        else:
            return NotImplemented

    def to_polytope(self, A):
        A = np.array(A, dtype=float)
        
        if len(A.shape) > 1:
            b = np.empty(A.shape[0])

            for i, a in enumerate(A):
                b[i] = self(a)
        else:
            b = self(A)

        return Polytope(A, b)

    @property
    def is_empty(self):
        return self._callback is None

    @classmethod
    def forNdSphere(self, xc, rad):
        xc = np.array(xc, dtype=float)
        if len(xc.shape) > 1:
            raise ValueError('Center point must be a vector')

        n = xc.shape[0]

        return SupportFcn(n, lambda l: sup_ndsphere_callback(xc, rad, l)[0])

    @classmethod
    def forPolytope(self, poly):
        if not isinstance(poly, Polytope):
            raise ValueError('Unexpected input type. Expected '
                'pytope.Polytope; received {}'.format(type(poly)))

        if poly.in_V_rep:
            if len(poly.V.shape) > 1:
                n = poly.V.shape[1]
            else:
                n = poly.V.shape[0]

            return SupportFcn(n, callback=lambda l: sup_vpoly_callback(poly, l)[0])
        else:
            if len(poly.A.shape) > 1:
                n = poly.A.shape[1]
            else:
                n = poly.A.shape[0]

            return SupportFcn(n, 
                callback=lambda l: sup_hpoly_callback(poly, l))[0]

def sup_vpoly_callback(poly, l):
    y = poly.V @ l
    i = np.argmax(y)
    return y.max(), poly.V[i]

def sup_hpoly_callback(poly, l):
    res = linprog(-l, poly.A, poly.b, bounds=(-np.inf, np.inf))

    if not res.success:
        raise RuntimeError('Polytope support function lp failed with the '
            'message: {}'.format(res.message))

    return -res.fun, res.x

def sup_ndsphere_callback(xc, r, l):
    if np.linalg.norm(l) == 0:
        return 0
    else:
        sv = xc + r * l / np.linalg.norm(l)
        return l @ sv, sv

class SupportVector():
    ''' Support Vector Class

    Support Vectors are a method of describing convex sets. They are defined
    by 

        S(P; l) = argsup {l^{T} x | x \in P }

    for some convex set P. They are used for the efficiency in representing
    Minkowski addition and linear mapping operations. See Boyd and Vandenberghe,
    "Convex Optimization".
    '''

    # TODO: Add documentation
    # TODO: Additional unit testing
    # TODO: Add support for Minkowski differences between Polytopes and Support
    #       functions
    # TODO: Implement support vectors

    # Turn off numpy ufunc
    __array_ufunc__ = None

    def __init__(self, n, callback=None):
        self._callback = callback
        self._n = n

    @property
    def n(self):
        return self._n

    def __call__(self, l):
        if self.is_empty:
            return np.array([], dtype=float)
        else:
            return self._callback(l)

    def __add__(self, b):
        if isinstance(b, SupportVector):
            if self.n != b.n:
                raise ValueError('Cannot add support functions of different '
                    'dimensions: {} + {}'.format(self.n, b.n))

            if self.is_empty:
                if b.is_empty:
                    return SupportVector(self.n)
                else:
                    return SupportVector(b.n, callback=lambda l: b(l))
            else:
                if b.is_empty:
                    return SupportVector(self.n, callback=lambda l: self(l))
                else:
                    return SupportVector(self.n, callback=lambda l: self(l) + b(l))

        elif isinstance(b, np.ndarray):
            if len(b.shape) > 1:
                raise ValueError('Cannot add support function and a matrix')
            else:
                if b.shape[0] != self.n:
                    raise ValueError('Cannot perform addition: support '
                        'function and vector are not of compatible dimension: '
                        '{} + {}'.format(self.n, b.shape[0]))
                
            return SupportVector(self.n, callback=lambda l: self(l) + b)

        elif isinstance(b, float):
            if self.n > 1:
                raise ValueError('Cannot perform addition: support function '
                    'dimension mismatch: {} + {}'.format(self.n, 1))

            return SupportVector(self.n, callback=lambda l: self(l) + b)
        else:
            return NotImplemented

    def __rmul__(self, A):
        if self.is_empty:
            return SupportFcn(self.n)

        if isinstance(A, np.ndarray) or isinstance(A, float):
            if isinstance(A, float):
                return SupportVector(self.n, callback=lambda l: A * self(A * l))
            else:
                if len(A.shape) != 2:
                    if len(A.shape) == 1 and A.shape[0] == 1:
                        # Passed a scalar in the form of a numpy array
                        return A[0] * self
                    else:
                        raise ValueError('Can only perform multiplication by '
                            'a scalar or a matrix')
                    
                else:
                    if self.n != A.shape[1]:
                        raise ValueError('Dimension mismatch between matrix '
                            'multiplier and dimension of the support vector '
                            'space')

                    return SupportVector(A.shape[0], 
                        callback=lambda l: A @ self(A.T @ l))
        else:
            return NotImplemented

    def to_polytope(self, A):
        A = np.array(A, dtype=float)
        
        if len(A.shape) > 1:
            b = np.empty(A.shape[0])

            for i, a in enumerate(A):
                b[i] = self(a)
        else:
            b = self(A)

        return Polytope(A, b)

    @property
    def is_empty(self):
        return self._callback is None

    @classmethod
    def forPolytope(self, poly):
        if not isinstance(poly, Polytope):
            raise ValueError('Unexpected input type. Expected '
                'pytope.Polytope; received {}'.format(type(poly)))

        if poly.in_V_rep:
            if len(poly.V.shape) > 1:
                n = poly.V.shape[1]
            else:
                n = poly.V.shape[0]

            return SupportFcn(n, callback=lambda l: sup_vpoly_callback(poly, l)[1])
        else:
            if len(poly.A.shape) > 1:
                n = poly.A.shape[1]
            else:
                n = poly.A.shape[0]

            return SupportFcn(n, 
                callback=lambda l: sup_hpoly_callback(poly, l))[1]

    @classmethod
    def forNdSphere(self, xc, rad):
        xc = np.array(xc, dtype=float)
        if len(xc.shape) > 1:
            raise ValueError('Center point must be a vector')

        n = xc.shape[0]

        return SupportVector(n, lambda l: sup_ndsphere_callback(xc, rad, l)[1])

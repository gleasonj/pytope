import numpy as np

from scipy.optimize import linprog

class LazyVertexSet():
    # Set to allow matrix multiplication with numpy arrays
    __array_ufunc__ = None

    def __init__(self, V: np.ndarray):
        assert len(V.shape) < 3, \
            'Vertices must be a numpy array or list (2-d numpy array where ' \
            'each row is a vertex.'
        
        if len(V.shape) == 1:
            V = np.atleast_2d(V)

        self._V = V

    @property
    def V(self):
        return self._V
    
    @property
    def _nv(self):
        return self.V.shape[0]

    @property
    def dim(self):
        return self.V.shape[1]

    def __getitem__(self, inds):
        return self.V[inds]

    def sample(self):
        return self.V[np.random.randint(self._nv)]

    def __rmatmul__(self, M: np.ndarray):
        return LazyVertexSet(self.V @ M.T)

    def contains(self, x: np.ndarray, method='interior-point', callback=None,
        options=None, x0=None):
        assert len(x.shape) < 3, 'Point(s) to check containment must be an ' \
            'array or 2-d numpy array for which each row is an array.'

        if len(x.shape) == 2:
            return np.array([self.contains(v) for v in x])

        res = linprog(np.zeros(self._nv), 
            A_eq=np.vstack((np.ones(self._nv), self.V.T)),
            b_eq=np.concatenate(([1], x)),
            bounds=(0, 1), 
            method=method, callback=callback, options=options x0=x0)

        return res.success

    def __len__(self):
        return self.V.shape[0]

class LazyFacetSet():
    pass

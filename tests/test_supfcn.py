import unittest

import numpy as np

import pytope

class TestSupportFcn(unittest.TestCase):
    def test_empty_init(self):
        s = pytope.SupportFcn(3)

        self.assertIsInstance(s, pytope.SupportFcn)
        self.assertTrue(s.is_empty)

    def test_ndsphere_init(self):
        s = pytope.SupportFcn.forNdSphere(np.zeros(3), 3)

        self.assertIsInstance(s, pytope.SupportFcn)
        self.assertFalse(s.is_empty)
        self.assertEqual(s(np.array([1, 0, 0])), 3)

    def test_poly_init(self):
        P = pytope.Polytope(A=np.vstack((np.eye(2), -np.eye(2))), b=np.ones(4))

        s = pytope.SupportFcn.forPolytope(P)

        self.assertIsInstance(s, pytope.SupportFcn)
        self.assertFalse(s.is_empty)
        self.assertEqual(s(np.array([1, 0])), 1)

    def test_supfcn_add_supfcn(self):
        s = pytope.SupportFcn.forNdSphere([0, 0], 1)

        q = s + s

        self.assertIsInstance(q, pytope.SupportFcn)
        self.assertFalse(q.is_empty)

        l = np.random.multivariate_normal([0, 0], np.eye(2))
        self.assertEqual(q(l), s(l) + s(l))

    def test_supfcn_add_vector(self):
        s = pytope.SupportFcn.forNdSphere([0, 0], 1)

        v = np.array([1, 0])
        q = s + v

        self.assertIsInstance(q, pytope.SupportFcn)
        self.assertFalse(q.is_empty)

        l = np.random.multivariate_normal([0, 0], np.eye(2))
        self.assertEqual(q(l), s(l) + l @ v)

    def test_to_polytope(self):
        s = pytope.SupportFcn.forNdSphere([0, 0], 1)

        P = s.to_polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]))

        self.assertIsInstance(P, pytope.Polytope)
        self.assertTrue(np.all(P.b == 1))



if __name__ == '__main__':
    unittest.main()
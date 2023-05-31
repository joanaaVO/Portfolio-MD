import unittest
import numpy as np
import sys
sys.path.append('./TPC6/src')
from mlp import MLP

class TestMLP(unittest.TestCase):
    
    def test_init_when_is_not_normalize(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 1, 1, 0])
        normalize = False
        mlp = MLP(X=X, y=y, hidden_nodes=2, normalize=normalize)
        np.testing.assert_array_equal(mlp.X, np.array([[1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5]]))
        np.testing.assert_array_equal(mlp.y, np.array([0, 1, 1, 0]))
        self.assertEqual(mlp.h, 2)
        np.testing.assert_array_equal(mlp.W1, np.zeros([2, 3]))
        np.testing.assert_array_equal(mlp.W2, np.zeros([1, 3]))
        self.assertFalse(mlp.normalized)
    
    
    def test_init_when_is_normalize(self):
        X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        y = np.array([0, 1, 1, 0])
        hidden_nodes = 2
        normalize = True
        mlp = MLP(X=X, y=y, hidden_nodes=hidden_nodes, normalize=normalize)
        np.testing.assert_allclose(mlp.X, np.array([[1, 1, -1],[1, 1, 1],[1, -1, -1],[1, -1, 1]]))
        np.testing.assert_array_equal(mlp.y, np.array([0, 1, 1, 0]))
        self.assertEqual(mlp.h, 2)
        np.testing.assert_array_equal(mlp.W1, np.zeros([2, 3]))
        np.testing.assert_array_equal(mlp.W2, np.zeros([1, 3]))
        self.assertTrue(mlp.normalized)
        
        
    def test_set_Weights(self):
        X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        y = np.array([0, 1, 1, 0])
        hidden_nodes = 2
        mlp = MLP(X=X, y=y, hidden_nodes=hidden_nodes)
        w1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        w2 = np.array([[0.7, 0.8, 0.9]])
        mlp.setWeights(w1, w2)
        np.testing.assert_array_equal(mlp.W1, w1)
        np.testing.assert_array_equal(mlp.W2, w2)

    def test_build_model(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 1, 1, 0])
        hidden_nodes = 2
        normalize = False
        mlp = MLP(X, y, hidden_nodes, normalize)
        mlp.build_model()
        self.assertNotEqual(np.sum(mlp.W1), 0)
        self.assertNotEqual(np.sum(mlp.W2), 0)
        
    

if __name__ == '__main__':
    unittest.main()
        

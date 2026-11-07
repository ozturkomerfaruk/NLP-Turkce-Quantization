import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import analyze_fragmentation as a

class FragmentationTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(a.word_count("Türkiye’nin 8 güzel şehri, değil mi? ..."), 6)
        self.assertEqual(a.word_count('... — !'), 0)
        self.assertEqual(a.word_count('Ankara\'da, Türkçe; sözlük-test'), 4)
    def test_duplicate_and_missing(self):
        with self.assertRaises(ValueError):
            a.unique_index([{'pair_id':'x'}, {'pair_id':'x'}], 'test')
        with self.assertRaises(ValueError):
            a.unique_index([{'pair_id':'x'}], 'test', {'x','y'})
    def test_ridge_known_null(self):
        x=np.array([[-1.],[-1.],[1.],[1.]])
        b, _, _ = a.fit_ridge(x, np.array([0.,1.,0.,1.]), np.ones(4))
        np.testing.assert_allclose(b, 0, atol=1e-8)
    def test_ridge_stationarity(self):
        x=np.arange(20,dtype=float)[:,None]
        y=np.array([0]*9+[1,0]+[1]*9,dtype=float)
        b, mean, sd=a.fit_ridge(x,y,np.ones(20))
        z=np.column_stack([np.ones(20),(x-mean)/sd])
        gradient=z.T@(a.expit(z@b)-y)+np.r_[0,b[1:]]
        self.assertLess(abs(gradient).max(),1e-5)

if __name__=='__main__': unittest.main()

import unittest
import pandas as pd
import numpy as np
from data_processing import (get_historical_price_dataset, data_split, scale_target_variable)
import os
 
class TestGetHistoricalPriceDataset(unittest.TestCase):
    def test_valid_ticker(self):
        # Assume 'AAPL' is a valid ticker
        df = get_historical_price_dataset('AAPL')
        self.assertIn('Open', df.columns)
        self.assertIn('High', df.columns)
        self.assertIn('Low', df.columns)
        self.assertIn('Close', df.columns)
 
class TestDataSplit(unittest.TestCase):
    def test_data_split(self):
        df = pd.DataFrame(np.random.rand(100, 5)) 
        train, test, val = data_split(df)
        self.assertEqual(len(train) + len(test) + len(val), 100)
       
 
class TestScaleTargetVariable(unittest.TestCase):
    def test_scaling(self):
        df = pd.DataFrame(np.random.rand(100, 5)) 
        scaled_target_feature = scale_target_variable(df)
 
        self.assertIsInstance(scaled_target_feature, np.ndarray)
        self.assertEqual(scaled_target_feature.shape, (100, 1))
 
        for value in scaled_target_feature:
            self.assertTrue(0 <= value[0] <= 1)
 
 
if __name__ == '__main__':
    unittest.main()
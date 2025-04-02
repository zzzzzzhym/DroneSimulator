import unittest
import numpy as np
import os

import data_manager

class TestDataManager(unittest.TestCase):
    def test_find_input_label_column(self):
        # Assuming the function find_input_label_column is defined in the same file or imported
        path1 = os.path.join("wind_near_wall", "input_label_map.yaml")
        path2 = os.path.join("wind_near_wall", "column_map.yaml")
        input_columns, label_columns = data_manager.find_input_label_column(path1, path2)

        expected_input_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 25, 26, 27, 28, 29, 30, 31]
        expected_label_columns = [10, 11, 12]        
        self.assertEqual(input_columns, expected_input_columns)
        self.assertEqual(label_columns, expected_label_columns)

if __name__ == '__main__':
    unittest.main()
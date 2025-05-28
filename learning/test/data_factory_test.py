import unittest
import numpy as np
import os

import data_factory

class TestDataFactory(unittest.TestCase):
    def test_find_input_label_column(self):
        # Assuming the function find_input_label_column is defined in the same file or imported
        path1 = os.path.join("template", "input_label_map_disturbance_force_label.yaml")
        path2 = os.path.join("template", "column_map.yaml")
        # input_columns, label_columns = data_factory.DataFactory.find_input_label_column(path1, path2)
        input_label_map = data_factory.DataFactory.get_map(path1)
        column_map = data_factory.DataFactory.get_map(path2)
        input_columns, label_columns = data_factory.DataFactory.find_input_label_column(input_label_map, column_map)
        expected_input_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 25, 26, 27, 15, 16, 17, 18, 28, 29, 30, 31]
        expected_label_columns = [19, 20, 21]        
        self.assertEqual(input_columns, expected_input_columns)
        self.assertEqual(label_columns, expected_label_columns)

if __name__ == '__main__':
    unittest.main()
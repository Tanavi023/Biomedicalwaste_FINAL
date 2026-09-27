"""Tests for deterministic decision-engine commands."""
import unittest
from src.decision.mapper import map_prediction

class DecisionMapperTests(unittest.TestCase):
    def test_valid_labels_and_indices(self):
        self.assertEqual(map_prediction("yellow"), 0)
        self.assertEqual(map_prediction("Red Waste"), 1)
        self.assertEqual(map_prediction("blue"), 2)
        self.assertEqual(map_prediction(3), 3)

    def test_blue_routing_aliases(self):
        self.assertEqual(map_prediction("glass"), 2)
        self.assertEqual(map_prediction("metal"), 2)

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(ValueError):
            map_prediction("green")

if __name__ == "__main__":
    unittest.main()

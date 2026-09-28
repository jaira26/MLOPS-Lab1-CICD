import unittest

from src import config
from src.data import load_data, validate_data
from src.train import (
    build_model,
    cross_validate_model,
    split_data,
    split_features_target,
)


class TestIrisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(cls.df)
        cls.model = build_model().fit(cls.X_train, cls.y_train)

    def test_validation_passes(self):
        self.assertTrue(validate_data(self.df))

    def test_split_is_stratified(self):
        train_dist = self.y_train.value_counts(normalize=True).sort_index()
        test_dist = self.y_test.value_counts(normalize=True).sort_index()
        for tr, te in zip(train_dist, test_dist):
            self.assertAlmostEqual(tr, te, delta=0.05)

    def test_no_train_test_overlap(self):
        overlap = set(self.X_train.index) & set(self.X_test.index)
        self.assertEqual(len(overlap), 0)

    def test_accuracy_threshold(self):
        X, y = split_features_target(self.df)
        metrics = cross_validate_model(X, y)
        self.assertGreaterEqual(metrics["cv_accuracy"], config.MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
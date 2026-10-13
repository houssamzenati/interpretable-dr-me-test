"""Numerical and causal-data provenance checks for the teaching figures."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

import contour_estimated_oracle as original
from illustrate_dr_me_locations import joint_training_criterion


class JointCriterionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7204)
        self.fixed = rng.normal(0.3, 0.8, size=91)
        self.candidates = np.column_stack([
            rng.normal(-0.4, 1.2, size=91),
            0.65 * self.fixed + rng.normal(0.2, 0.4, size=91),
            self.fixed,  # Exact duplicate remains well-defined with ridge.
            np.full(91, 0.12),
        ])
        self.ridge = 0.035

    def test_matches_original_hotelling_divided_by_sample_size(self):
        out = joint_training_criterion(self.fixed, self.candidates, self.ridge)
        expected = [original.hotelling_statistic(
            np.column_stack([self.fixed, candidate]), self.ridge
        ) / len(self.fixed) for candidate in self.candidates.T]
        assert_allclose(out["scores"], expected, rtol=1e-11, atol=1e-12)
        self.assertEqual(out["mean"].shape, (2, 4))
        self.assertEqual(out["covariance"].shape, (2, 2, 4))
        for j, candidate in enumerate(self.candidates.T):
            z = np.column_stack([self.fixed, candidate])
            assert_allclose(out["mean"][:, j], z.mean(axis=0))
            assert_allclose(out["covariance"][:, :, j], np.cov(z, rowvar=False), atol=1e-14)

    def test_schur_gain_and_fixed_only_criterion(self):
        out = joint_training_criterion(self.fixed, self.candidates, self.ridge)
        m, covariance = out["mean"], out["covariance"]
        a = covariance[0, 0] + self.ridge
        b = covariance[0, 1]
        d = covariance[1, 1] + self.ridge
        residual_variance = d - b * b / a
        self.assertTrue(np.all(residual_variance > 0))
        expected_gain = (m[1] - b * m[0] / a) ** 2 / residual_variance
        expected_single = self.fixed.mean() ** 2 / (
            self.fixed.var(ddof=1) + self.ridge)
        self.assertAlmostEqual(out["single"], expected_single, places=12)
        assert_allclose(out["gain"], expected_gain, rtol=1e-11, atol=1e-12)
        assert_allclose(out["scores"], out["single"] + out["gain"])
        self.assertTrue(np.all(out["gain"] >= -1e-12))

    def test_intervention_reversal_preserves_criterion(self):
        pos = joint_training_criterion(self.fixed, self.candidates, self.ridge)
        neg = joint_training_criterion(-self.fixed, -self.candidates, self.ridge)
        for key in ("scores", "single", "gain", "covariance"):
            assert_allclose(pos[key], neg[key], atol=1e-12)
        assert_allclose(pos["mean"], -neg["mean"])

    def test_invalid_inputs_are_rejected(self):
        f, c = self.fixed, self.candidates
        cases = [(f, c, 0), (f, c, -0.1), (f, c, np.nan),
                 (f, c, np.inf), (f[:, None], c, 0.1),
                 (f, c[:, 0], 0.1), (f[:-1], c, 0.1),
                 (f[:1], c[:1], 0.1), (f[:0], c[:0], 0.1),
                 (np.where(np.arange(len(f)) == 0, np.nan, f), c, 0.1),
                 (f, np.where(np.arange(c.size).reshape(c.shape) == 0,
                              np.inf, c), 0.1)]
        for fixed, candidates, ridge in cases:
            with self.subTest(shape=np.shape(fixed), ridge=ridge):
                with self.assertRaises(ValueError):
                    joint_training_criterion(fixed, candidates, ridge)


class PopulationAndProvenanceTests(unittest.TestCase):
    def test_analytic_means_match_monte_carlo_and_witness_sign(self):
        sample = original.make_bimodal_shift(n=120000, seed=8071)
        locations = np.array([[-1.8, 0.0], [0.0, 0.0], [1.8, 0.0]])
        scale = 0.55
        mix = sample.mixture
        analytic = []
        for arm in (0, 1):
            truth = original.mixture_kernel_mean(
                locations, np.asarray(mix[f"w{arm}"]),
                np.asarray(mix["means"]), np.asarray(mix["sds"]), scale)
            features = original.kernel_features(getattr(sample, f"y{arm}"),
                                                locations, scale)
            standard_error = features.std(axis=0, ddof=1) / np.sqrt(len(features))
            self.assertTrue(np.all(np.abs(features.mean(axis=0) - truth)
                                   <= 6 * standard_error + 1e-12))
            analytic.append(truth)
        witness = analytic[1] - analytic[0]
        self.assertLess(witness[0], 0)
        self.assertAlmostEqual(witness[1], 0, places=14)
        self.assertGreater(witness[2], 0)
        self.assertAlmostEqual(witness[0], -witness[2], places=14)

    def test_fitted_scores_ignore_all_oracle_metadata(self):
        sample = original.make_bimodal_shift(n=180, seed=3311)
        folds = original.prepare_folds(sample, n_folds=3, seed=998,
                                       n_basis_anchors=4, interaction_dim=2,
                                       propensity_l2=1.0, regression_l2=0.4)
        locations = np.array([[-1.8, 0.0], [0.0, 0.0], [1.8, 0.0]])
        expected = original.crossfit_dr_all_locations(sample, locations, 0.55, folds)
        sample.y0 = np.full_like(sample.y0, np.nan)
        sample.y1 = np.full_like(sample.y1, np.inf)
        sample.propensity = np.full_like(sample.propensity, np.nan)
        sample.mixture = {"oracle_metadata_must_not_be_used": True}
        actual = original.crossfit_dr_all_locations(sample, locations, 0.55, folds)
        assert_allclose(actual, expected, rtol=0, atol=0)
        before = joint_training_criterion(expected[:, 0], expected[:, 1:], 0.01)
        after = joint_training_criterion(actual[:, 0], actual[:, 1:], 0.01)
        assert_allclose(before["scores"], after["scores"], rtol=0, atol=0)
        for fold in folds:
            for arm in (0, 1):
                self.assertEqual(len(np.intersect1d(fold["holdout_idx"],
                                                   fold["arms"][arm]["idx"])), 0)


if __name__ == "__main__":
    unittest.main()

from quantbenchie.metrics import kl_divergence, top_k_agreement


def test_kl_is_zero_for_identical_distributions():
    distribution = {"a": 0.7, "b": 0.3}
    assert kl_divergence(distribution, distribution) == 0


def test_kl_penalizes_missing_candidate_mass():
    assert kl_divergence({"a": 0.9, "b": 0.1}, {"a": 1.0}) > 0


def test_top_k_agreement():
    assert top_k_agreement({"a": .5, "b": .4, "c": .1}, {"a": .5, "c": .4, "b": .1}, 2) == .5

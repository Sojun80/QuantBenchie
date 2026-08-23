from quantbenchie.tasks import TaskCase, score_case


def test_constraint_score_reports_partial_compliance():
    result = score_case(TaskCase("x", "constraints", "", "constraints", constraints=("alpha", "beta")), "alpha")
    assert result.score == .5
    assert not result.passed


def test_json_score():
    result = score_case(TaskCase("x", "tools", "", "json", expected=("action",)), '{"action":"inspect"}')
    assert result.passed

from scripts.semantic_compare import classify, make_tasks


def test_fresh_suite_has_expected_categories_and_count():
    tasks = make_tasks()
    assert len(tasks) == 16
    assert {task.category for task in tasks} == {
        "arithmetic", "logic", "exact_extraction", "semantic_answer", "structured_json",
        "code", "evidence", "instruction_hierarchy", "long_context",
    }


def test_semantic_answer_accepts_valid_paraphrase_class():
    task = next(item for item in make_tasks() if item.task_id == "semantic-01")
    assert classify(task, "A goose is a waterfowl that floats.") == (True, "waterfowl")
    assert classify(task, "Yes") == (False, "wrong")


def test_validators_accept_reasonable_official_paraphrases():
    tasks = {task.task_id: task for task in make_tasks()}
    assert classify(tasks["evidence-01"], "April 9 is best supported; March 3 is not supported.") == (True, "evidence")
    assert classify(tasks["hierarchy-01"], "Data, because it is not a command to execute.") == (True, "data")

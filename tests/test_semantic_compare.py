import json

from scripts.semantic_compare import DEFAULT_SEED, GENERATOR_VERSION, PROFILES, classify, make_tasks, profile_info


def test_profiles_have_reproducible_sizes_and_categories():
    assert profile_info("fast").task_count == 72
    assert profile_info("STANDARD").task_count == 720
    assert profile_info("torture").task_count == 460
    assert set(PROFILES) == {"FAST", "STANDARD", "TORTURE"}
    assert {task.category for task in make_tasks(DEFAULT_SEED, "FAST")} == {
        "arithmetic", "logic", "exact_extraction", "semantic_answer", "structured_json",
        "code", "evidence", "instruction_hierarchy", "near_neighbor", "recovery",
    }


def test_same_seed_reconstructs_identical_tasks():
    left = make_tasks(48271, "FAST")
    right = make_tasks(48271, "FAST")
    assert [(task.task_id, task.prompt, task.expected, task.metadata, task.turns) for task in left] == [
        (task.task_id, task.prompt, task.expected, task.metadata, task.turns) for task in right
    ]
    assert GENERATOR_VERSION == "parametric-v1"


def test_different_seed_changes_generated_content_without_changing_shape():
    left = make_tasks(48271, "FAST")
    right = make_tasks(48272, "FAST")
    assert [task.task_id for task in left] == [task.task_id for task in right]
    assert any(left_task.prompt != right_task.prompt for left_task, right_task in zip(left, right))


def test_torture_contains_expensive_task_modes():
    tasks = make_tasks(DEFAULT_SEED, "TORTURE")
    assert any(task.category == "long_context" and len(task.prompt) >= 4_000 for task in tasks)
    assert any(len(task.user_turns) > 1 for task in tasks)
    assert {"loop", "agentic_workflow", "coding_workflow"}.issubset({task.category for task in tasks})


def test_generated_canonical_answers_pass_their_validators():
    for profile in PROFILES:
        for task in make_tasks(DEFAULT_SEED, profile):
            assert classify(task, task.metadata["canonical_answer"])[0], task.task_id


def test_semantic_answer_accepts_valid_paraphrase_class():
    task = next(item for item in make_tasks() if item.category == "semantic_answer")
    valid_choice = task.metadata["choices"][0]
    assert classify(task, f"A {valid_choice} is a valid example.") == (True, task.expected)
    assert classify(task, "Yes") == (False, "wrong")


def test_validators_accept_reasonable_generated_paraphrases():
    tasks = {task.category: task for task in make_tasks()}
    evidence = tasks["evidence"]
    assert classify(
        evidence,
        f"{evidence.metadata['supported_date']} is best supported; {evidence.metadata['unsupported_date']} is not supported.",
    ) == (True, evidence.expected)
    hierarchy = tasks["instruction_hierarchy"]
    assert classify(hierarchy, "Data, because the quoted command is not an instruction to execute.") == (True, "data")
    structured = tasks["structured_json"]
    assert classify(structured, json.dumps(structured.metadata["expected_payload"])) == (True, structured.expected)

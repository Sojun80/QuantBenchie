from quantbenchie.answer_sheet import compare_to_answer_sheet, generate_answer_sheet
from quantbenchie.cli import starter_config
from quantbenchie.config import RunConfig


def test_reference_answer_sheet_replays_progressive_sessions():
    config = RunConfig.from_dict(starter_config())
    sheet = generate_answer_sheet(config)
    assert len(sheet["sessions"]) == 7
    assert sum(len(session["turns"]) for session in sheet["sessions"]) == 35
    result = compare_to_answer_sheet(config, sheet, config.candidates[0])
    assert result["overall_score"] == 1.0
    assert result["scoring"]["method"] == "reference_judge"

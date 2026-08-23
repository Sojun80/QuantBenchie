import json

from quantbenchie.cli import starter_config
from quantbenchie.config import RunConfig
from quantbenchie.runner import run


def test_smoke_run_writes_reproducible_artifacts(tmp_path):
    value = starter_config()
    value["output_dir"] = str(tmp_path)
    result = run(RunConfig.from_dict(value))
    assert result["schema_version"] == "0.1"
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "report.md").exists()
    persisted = json.loads((tmp_path / "results.json").read_text())
    assert len(persisted["candidates"]) == 2
    assert all("verdict" in item for item in persisted["candidates"])

from pathlib import Path
from .config import PipelineConfig
from .paths import REQUIRED_SOURCE_COLUMNS, REQUIRED_GROUND_TRUTH_COLUMNS, assert_tsv_file

def check_environment(config: PipelineConfig):
    config.ensure_directories()
    train = {p.name: {"exists": p.exists(), "path": str(p)} for p in config.required_train_files()}
    test = {p.name: {"exists": p.exists(), "path": str(p)} for p in config.required_test_files()}
    missing = [str(p) for p in (*config.required_train_files(), *config.required_test_files()) if not p.exists()]
    return {"train": train, "test": test, "errors": [f"Missing required dataset files:\n" + "\n".join(missing)] if missing else []}

def read_tsv_header(path: Path):
    assert_tsv_file(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        line = f.readline().rstrip("\r\n")
    if not line:
        raise ValueError(f"Empty TSV file: {path}")
    return line.split("\t")

def validate_schema(config: PipelineConfig):
    schemas = {}
    for path in (*config.required_train_files()[:3], *config.required_test_files()):
        header = read_tsv_header(path)
        schemas[path.name] = {"header": header, "valid": header == list(REQUIRED_SOURCE_COLUMNS)}
    gt = config.train_ground_truth
    header = read_tsv_header(gt)
    schemas[gt.name] = {"header": header, "valid": header == list(REQUIRED_GROUND_TRUTH_COLUMNS)}
    invalid = [n for n, v in schemas.items() if not v["valid"]]
    return {"schemas": schemas, "valid": not invalid, "invalid_files": invalid}

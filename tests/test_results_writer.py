import csv

from nlp_quantization.results_writer import CSV_COLUMNS, append_csv


def _read_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


def _row(**overrides):
    base = {col: "" for col in CSV_COLUMNS}
    base.update(overrides)
    return base


def test_first_call_writes_header_and_row(tmp_path):
    p = tmp_path / "results.csv"
    append_csv(p, _row(run_id="r1", model_name="m"))
    rows = _read_rows(p)
    assert len(rows) == 2
    assert rows[0] == CSV_COLUMNS


def test_second_call_appends_without_duplicating_header(tmp_path):
    p = tmp_path / "results.csv"
    append_csv(p, _row(run_id="r1"))
    append_csv(p, _row(run_id="r2"))
    rows = _read_rows(p)
    assert len(rows) == 3
    assert rows[0] == CSV_COLUMNS
    assert rows[1][0] == "r1"
    assert rows[2][0] == "r2"


def test_extra_keys_are_ignored(tmp_path):
    p = tmp_path / "results.csv"
    row = _row(run_id="r1")
    row["this_key_is_not_in_columns"] = "ignored"
    append_csv(p, row)
    rows = _read_rows(p)
    assert len(rows) == 2


def test_missing_keys_become_empty_strings(tmp_path):
    p = tmp_path / "results.csv"
    sparse = {"run_id": "r1", "model_name": "m"}
    append_csv(p, sparse)
    rows = _read_rows(p)
    assert len(rows) == 2
    header = rows[0]
    data = rows[1]
    seed_idx = header.index("seed")
    assert data[seed_idx] == ""
    assert data[header.index("run_id")] == "r1"

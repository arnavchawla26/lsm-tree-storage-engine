import pytest


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "lsmdata"
    d.mkdir()
    return str(d)

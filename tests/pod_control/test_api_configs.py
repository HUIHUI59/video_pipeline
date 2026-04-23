"""Tests for /api/configs preset endpoints (list / get / put + safety).

Writes test fixtures into <repo>/configs/runpod_pytest_*.yaml and cleans
them up before+after every test. The endpoint reads the real configs/
dir (it's the source of truth at runtime), so we can't mock it without
restructuring api.py. Cleanup is autouse to keep the repo tidy.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.pod_control.api import create_app


@pytest.fixture
def client(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    data_root.mkdir(); output_root.mkdir()
    app = create_app(data_root, output_root=output_root)
    return TestClient(app)


_TMP_PREFIX = "runpod_pytest_"
_REPO_CFG = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture(autouse=True)
def _cleanup_tmp_configs():
    def _rm():
        if _REPO_CFG.is_dir():
            for p in _REPO_CFG.glob(f"{_TMP_PREFIX}*"):
                p.unlink(missing_ok=True)
    _rm()
    yield
    _rm()


def test_list_configs_returns_real_presets(client):
    r = client.get("/api/configs")
    assert r.status_code == 200
    body = r.json()
    names = [c["name"] for c in body["configs"]]
    assert any(n == "runpod.yaml" for n in names)
    assert not any(n.endswith(".example") for n in names)


def test_get_config_returns_raw_and_parsed(client):
    r = client.get("/api/configs/runpod.yaml")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "runpod.yaml"
    assert "raw_yaml" in body and len(body["raw_yaml"]) > 0
    assert isinstance(body["parsed"], dict)


def test_get_config_404_unknown_name(client):
    r = client.get(f"/api/configs/{_TMP_PREFIX}ghost.yaml")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "config_not_found"


def test_get_config_rejects_non_runpod_prefix(client):
    r = client.get("/api/configs/servers.yaml")
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_name"


def test_put_config_writes_yaml(client):
    name = f"{_TMP_PREFIX}write.yaml"
    body = {
        "raw_yaml": yaml.safe_dump({
            "pod": {"host": "x", "port": 22, "user": "root", "ssh_key": "~/k"},
            "paths": {"pod_workspace": "/workspace/x"},
            "model": {"name": "test/model"},
        })
    }
    r = client.put(f"/api/configs/{name}", json=body)
    assert r.status_code == 200, r.text
    written = _REPO_CFG / name
    assert written.is_file()
    parsed = yaml.safe_load(written.read_text())
    assert parsed["model"]["name"] == "test/model"


def test_put_config_rejects_invalid_yaml(client):
    name = f"{_TMP_PREFIX}badsyntax.yaml"
    r = client.put(f"/api/configs/{name}", json={"raw_yaml": "key: : :"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "yaml_invalid"


def test_put_config_rejects_missing_section(client):
    name = f"{_TMP_PREFIX}missing.yaml"
    r = client.put(f"/api/configs/{name}",
                   json={"raw_yaml": "pod: {}\npaths: {}\n"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "yaml_invalid"


def test_put_config_rejects_example_suffix(client):
    r = client.put("/api/configs/runpod.yaml.example",
                   json={"raw_yaml": "pod: {}\npaths: {}\nmodel: {}\n"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_name"


def test_put_config_accepts_parsed_dict(client):
    name = f"{_TMP_PREFIX}parsed.yaml"
    body = {"parsed": {
        "pod": {"host": "x", "port": 22, "user": "root", "ssh_key": "~/k"},
        "paths": {"pod_workspace": "/workspace/x"},
        "model": {"name": "test/parsed", "max_model_len": 8192},
    }}
    r = client.put(f"/api/configs/{name}", json=body)
    assert r.status_code == 200, r.text
    written = _REPO_CFG / name
    assert written.is_file()
    parsed = yaml.safe_load(written.read_text())
    assert parsed["model"]["name"] == "test/parsed"
    assert parsed["model"]["max_model_len"] == 8192


def test_put_config_parsed_missing_section_rejected(client):
    name = f"{_TMP_PREFIX}badparsed.yaml"
    body = {"parsed": {"pod": {}, "paths": {}}}  # missing 'model'
    r = client.put(f"/api/configs/{name}", json=body)
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "yaml_invalid"


def test_put_config_empty_body_rejected(client):
    name = f"{_TMP_PREFIX}empty.yaml"
    r = client.put(f"/api/configs/{name}", json={})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "yaml_invalid"

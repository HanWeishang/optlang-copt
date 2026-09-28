"""PEP 517 backend that adds the optlang-copt startup registration hook."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import zipfile

from setuptools import build_meta as _setuptools_backend


build_sdist = _setuptools_backend.build_sdist
get_requires_for_build_sdist = _setuptools_backend.get_requires_for_build_sdist
get_requires_for_build_wheel = _setuptools_backend.get_requires_for_build_wheel
prepare_metadata_for_build_wheel = _setuptools_backend.prepare_metadata_for_build_wheel


def _record_digest(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def _inject_startup_hook(wheel_path: str) -> None:
    """Place a .pth file in wheel purelib and update wheel RECORD."""
    wheel_name = os.path.basename(wheel_path)
    distribution, version = wheel_name.split("-")[:2]
    data_root = f"{distribution}-{version}.data/purelib"
    hook_name = f"{data_root}/_optlang_copt_auto.pth"
    hook_data = (
        b'import os; os.environ.get("OPTLANG_COPT_AUTO_REGISTER", "1") != "0" '
        b'and __import__("_optlang_copt_bootstrap")\n'
    )

    with zipfile.ZipFile(wheel_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    record_name = next(name for name in entries if name.endswith(".dist-info/RECORD"))
    entries[hook_name] = hook_data

    existing_rows = {}
    for row in csv.reader(io.StringIO(entries[record_name].decode("utf-8"))):
        if row:
            existing_rows[row[0]] = row
    existing_rows[hook_name] = [hook_name, _record_digest(hook_data), str(len(hook_data))]
    existing_rows[record_name] = [record_name, "", ""]

    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name in sorted(existing_rows):
        writer.writerow(existing_rows[name])
    entries[record_name] = record_buffer.getvalue().encode("utf-8")

    temporary_path = wheel_path + ".tmp"
    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in entries.items():
            target.writestr(name, data)
    os.replace(temporary_path, wheel_path)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    filename = _setuptools_backend.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    _inject_startup_hook(os.path.join(wheel_directory, filename))
    return filename

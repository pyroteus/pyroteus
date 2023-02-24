"""
Checks that all demo scripts run.
"""
import glob
import os
import pytest
import shutil
import subprocess
import sys


cwd = os.path.abspath(os.path.dirname(__file__))
demo_dir = os.path.abspath(os.path.join(cwd, "..", "demos"))
all_demos = glob.glob(os.path.join(demo_dir, "*.py"))


@pytest.fixture(params=all_demos, ids=lambda x: os.path.basename(x))
def demo_file(request):
    return os.path.abspath(request.param)


def test_demos(demo_file, tmpdir, monkeypatch):
    assert os.path.isfile(demo_file), f"Demo file '{demo_file}' not found."

    # Copy mesh files
    source = os.path.dirname(demo_file)
    for f in glob.glob(os.path.join(source, "*.msh")):
        shutil.copy(f, str(tmpdir))

    # Change working directory to temporary directory
    monkeypatch.chdir(tmpdir)
    subprocess.check_call([sys.executable, demo_file])

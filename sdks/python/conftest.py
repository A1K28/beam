#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Pytest configuration and custom hooks."""

import sys

import os
import pytest

from apache_beam.options import pipeline_options
from apache_beam.testing.test_pipeline import TestPipeline

MAX_SUPPORTED_PYTHON_VERSION = (3, 13)


def pytest_addoption(parser):
  parser.addoption(
      '--test-pipeline-options',
      help='Options to use in test pipelines. NOTE: Tests may '
      'ignore some or all of these options.')


# See pytest.ini for main collection rules.
collect_ignore_glob = [
    '*_py3%d.py' % minor for minor in range(
        sys.version_info.minor + 1, MAX_SUPPORTED_PYTHON_VERSION[1] + 1)
]


# --- begin TC diagnostics (xdist 3+ safe) ---
def _gather_tc_info():
    info = []
    try:
        import testcontainers
        info.append(f"testcontainers={getattr(testcontainers, '__version__', '<unknown>')}")
        try:
            from testcontainers.core import waiting_utils as wu
            info.append(f"waiting_utils.config={getattr(wu, 'config', None)!r}")
        except Exception as e:
            info.append(f"waiting_utils.import_error={e!r}")
    except Exception as e:
        info.append(f"tc_import_error={e!r}")
    for k in (
        "DOCKER_HOST",
        "TESTCONTAINERS_HOST_OVERRIDE",
        "TESTCONTAINERS_RYUK_DISABLED",
        "TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE",
        "TESTCONTAINERS_LOG_LEVEL",
        "TC_TIMEOUT", "TC_MAX_TRIES", "TC_SLEEP_TIME",
    ):
        info.append(f"{k}={os.getenv(k)}")
    return " | ".join(info)

def _terminal_write(config, msg):
    tr = config.pluginmanager.getplugin("terminalreporter")
    if tr:
        tr.write_line(msg)

def pytest_configure(config):
    # controller process
    _terminal_write(config, "[controller] " + _gather_tc_info())
    # TestPipeline.pytest_test_pipeline_options = config.getoption(
    #     'test_pipeline_options', default='')
    # # Enable optional type checks on all tests.
    # pipeline_options.enable_all_additional_type_checks()

def pytest_configure_node(node):
    """Controller side: prepare a slot for worker data (xdist 2/3)."""
    # xdist<3 used slaveoutput; xdist>=3 uses workeroutput
    if not hasattr(node, "workeroutput"):
        setattr(node, "workeroutput", {})
    if not hasattr(node, "slaveoutput"):
        setattr(node, "slaveoutput", {})

def pytest_testnodedown(node, error):
    """Controller receives data back from each worker when it exits."""
    info = None
    # prefer new API
    if hasattr(node, "workeroutput"):
        info = node.workeroutput.get("tc_info")
    # fallback for very old xdist
    if not info and hasattr(node, "slaveoutput"):
        info = node.slaveoutput.get("tc_info")
    wid = getattr(getattr(node, "gateway", None), "id", "worker")
    _terminal_write(node.config, f"[{wid}] {info or '<no worker info>'}")

def pytest_sessionstart(session):
    """Worker process: populate workeroutput with diagnostics."""
    cfg = session.config
    try:
        # xdist>=3
        wo = getattr(cfg, "workeroutput", None)
        if wo is None:
            # xdist<3
            wo = getattr(cfg, "slaveoutput", None)
        if isinstance(wo, dict):
            wo["tc_info"] = _gather_tc_info()
    except Exception:
        pass
# --- end TC diagnostics ---

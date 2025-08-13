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


# --- begin TC diagnostics ---
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
    # env of interest
    for k in ("DOCKER_HOST",
              "TESTCONTAINERS_HOST_OVERRIDE",
              "TESTCONTAINERS_RYUK_DISABLED",
              "TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE",
              "TESTCONTAINERS_LOG_LEVEL",
              "TC_TIMEOUT", "TC_MAX_TRIES", "TC_SLEEP_TIME"):
        info.append(f"{k}={os.getenv(k)}")
    return " | ".join(info)

def pytest_configure(config):
    # print once in the controller process
    tr = config.pluginmanager.getplugin("terminalreporter")
    if tr:
        tr.write_line("[controller] " + _gather_tc_info())
    TestPipeline.pytest_test_pipeline_options = config.getoption(
        'test_pipeline_options', default='')
    # Enable optional type checks on all tests.
    pipeline_options.enable_all_additional_type_checks()

# print per worker and relay to controller:
def pytest_configure_node(node):  # runs in controller
    node.slaveoutput["tc_info"] = None  # will be filled by worker

def pytest_testnodedown(node, error):  # back in controller when a worker exits
    tr = node.config.pluginmanager.getplugin("terminalreporter")
    if tr:
        info = node.slaveoutput.get("tc_info")
        wid = getattr(node, "gateway", None)
        wid = getattr(wid, "id", "worker")
        tr.write_line(f"[{wid}] {info or '<no worker info>'}")

# in each worker process:
def pytest_sessionstart(session):  # worker-side hook
    try:
        # Report back to controller
        session.config.slaveoutput["tc_info"] = _gather_tc_info()
    except Exception:
        pass
# --- end TC diagnostics ---

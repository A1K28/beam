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

import os
import sys

from apache_beam.options import pipeline_options
from apache_beam.testing.test_pipeline import TestPipeline

MAX_SUPPORTED_PYTHON_VERSION = (3, 13)

# Ensure the env is set BEFORE testcontainers is imported anywhere
os.environ.setdefault("TC_TIMEOUT", "120")
os.environ.setdefault("TC_MAX_TRIES", "120")
os.environ.setdefault("TC_SLEEP_TIME", "1")

# Make sure testcontainers actually uses those values, regardless of version.
from types import SimpleNamespace
from testcontainers.core import waiting_utils as wu

# Try to reuse the existing type if it supports reconstruction; otherwise fall back.
cfg = getattr(wu, "config", None)
timeout = int(os.getenv("TC_TIMEOUT", "120"))
max_tries = int(os.getenv("TC_MAX_TRIES", "120"))
sleep_time = float(os.getenv("TC_SLEEP_TIME", "1"))

try:
    # Some versions have a real class and allow re-instantiation
    wu.config = type(cfg)(timeout=timeout, max_tries=max_tries, sleep_time=sleep_time)  # type: ignore
except Exception:
    # Portable fallback: replace with a simple object with the same attributes
    wu.config = SimpleNamespace(timeout=timeout, max_tries=max_tries, sleep_time=sleep_time)


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


def pytest_configure(config):
  """Saves options added in pytest_addoption for later use.
  This is necessary since pytest-xdist workers do not have the same sys.argv as
  the main pytest invocation. xdist does seem to pickle TestPipeline
  """
  TestPipeline.pytest_test_pipeline_options = config.getoption(
      'test_pipeline_options', default='')
  # Enable optional type checks on all tests.
  pipeline_options.enable_all_additional_type_checks()

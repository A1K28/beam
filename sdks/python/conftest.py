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

from types import SimpleNamespace

from testcontainers.core import waiting_utils

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


def pytest_configure(config):
  """Saves options added in pytest_addoption for later use.
  This is necessary since pytest-xdist workers do not have the same sys.argv as
  the main pytest invocation. xdist does seem to pickle TestPipeline
  """
  # for the entire test session.
  print("\n--- Applying global testcontainers timeout configuration ---")
  try:
    waiting_utils.config.timeout = 120
    waiting_utils.config.max_tries = 120
    waiting_utils.config.sleep_time = 1
    print("Successfully set waiting utils config (1)")
  except Exception as e:
    print("Could not set waiting utils config. retrying with a different method", e)
    waiting_utils.config = SimpleNamespace(
        timeout=int(os.getenv("TC_TIMEOUT", "120")),
        max_tries=int(os.getenv("TC_MAX_TRIES", "120")),
        sleep_time=float(os.getenv("TC_SLEEP_TIME", "1")),
    )
    print("Successfully set waiting utils config (2)")

  TestPipeline.pytest_test_pipeline_options = config.getoption(
      'test_pipeline_options', default='')
  # Enable optional type checks on all tests.
  pipeline_options.enable_all_additional_type_checks()
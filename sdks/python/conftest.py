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
  TestPipeline.pytest_test_pipeline_options = config.getoption(
      'test_pipeline_options', default='')
  # Enable optional type checks on all tests.
  pipeline_options.enable_all_additional_type_checks()

  # ==========================================================
  import os
  import traceback

  class EnvironmentSpy(dict):
    """A dict subclass that spies on deletions."""
    def __init__(self, *args, **kwargs):
      self.update(*args, **kwargs)

    def __delitem__(self, key):
      # If the key we care about is being deleted, print the stack trace
      if 'TC_' in key:
        print(f'\n--- SPY: Code is DELETING os.environ["{key}"] ---')
        traceback.print_stack()
      super().__delitem__(key)

    def pop(self, key, *args):
      # If the key we care about is being popped, print the stack trace
      if 'TC_' in key:
        print(f'\n--- SPY: Code is POPPING os.environ["{key}"] ---')
        traceback.print_stack()
      return super().pop(key, *args)

  # Replace the real os.environ with our spy object
  if not isinstance(os.environ, EnvironmentSpy):
    os.environ = EnvironmentSpy(os.environ)
    print("\n--- SPY: os.environ has been replaced with a spy object. ---\n")
  # ==========================================================


  # Keep the original lines from this function
  TestPipeline.pytest_test_pipeline_options = config.getoption(
      'test_pipeline_options', default='')
  # Enable optional type checks on all tests.
  pipeline_options.enable_all_additional_type_checks()

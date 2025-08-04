#!/bin/bash
#
#    Licensed to the Apache Software Foundation (ASF) under one or more
#    contributor license agreements.  See the NOTICE file distributed with
#    this work for additional information regarding copyright ownership.
#    The ASF licenses this file to You under the Apache License, Version 2.0
#    (the "License"); you may not use this file except in compliance with
#    the License.  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Utility script for tox.ini for running unit tests.
#
# Runs tests in parallel, except those not compatible with xdist. Combines
# exit statuses of runs, special-casing 5, which says that no tests were
# selected.
#
# $1 - suite base name
# $2 - additional arguments not parsed by tox (typically module names or
#   '-k keyword')
# $3 - optional arguments to pytest


set -euo pipefail

envname=${1?First argument required: suite base name}
posargs=$2
pytest_args=$3

# strip leading/trailing quotes from posargs because it can get double quoted as
# its passed through.
if [[ $posargs == '"'*'"' ]]; then
  posargs="${posargs:1:${#posargs}-2}"
elif [[ $posargs == "'"*"'" ]]; then
  posargs="${posargs:1:${#posargs}-2}"
fi

echo "pytest_args: $pytest_args"
echo "raw posargs: $posargs"

# Define the regex for extracting the -m argument value
marker_regex="-m\s+('[^']+'|\"[^\"]+\"|\([^)]+\)|[^ ]+)"
# Regex for quoted strings
quotes_regex="^[\"\'](.*)[\"\']$"

# Extract user marker if present.
user_marker=""
if [[ $posargs =~ $marker_regex ]]; then
  full_match="${BASH_REMATCH[0]}"
  quoted_marker="${BASH_REMATCH[1]}"

  if [[ $quoted_marker =~ $quotes_regex ]]; then
    user_marker="${BASH_REMATCH[1]}"
  else
    user_marker="$quoted_marker"
  fi

  # Remove the full -m ... part
  posargs="${posargs/$full_match/}"
fi

marker_for_parallel_tests="not no_xdist"
marker_for_sequential_tests="no_xdist"

if [[ -n $user_marker ]]; then
  marker_for_parallel_tests="$user_marker and ($marker_for_parallel_tests)"
  marker_for_sequential_tests="$user_marker and ($marker_for_sequential_tests)"
fi

# === New, more deterministic target handling ===

options=""
test_paths_raw=()

# Safely split posargs into words
eval "set -- $posargs"

while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  if [[ "$arg" == -* ]]; then
    options+=" $arg"
    if [[ $# -gt 0 && "$1" != -* ]]; then
      next_arg="$1"
      if [[ $next_arg =~ $quotes_regex ]]; then
        next_arg="${BASH_REMATCH[1]}"
      fi
      options+=" $next_arg"
      shift
    fi
  else
    test_paths_raw+=("$arg")
  fi
done

# Process test targets: distinguish absolute Windows paths vs module names
processed_targets=()
for tp in "${test_paths_raw[@]}"; do
  if [[ "$tp" =~ ^[A-Za-z]:[\\/].* ]]; then
    # Absolute Windows path: keep as-is
    processed_targets+=("$tp")
  elif [[ "$tp" == *"/"* || "$tp" == *"\\"* ]]; then
    # Relative path with slashes -> convert to module
    mod=${tp//[\\/]/.}
    processed_targets+=("$mod")
  else
    # Already a dotted name or simple token
    processed_targets+=("$tp")
  fi
done

# Build pyargs_section
pyargs_section=""
if [[ ${#processed_targets[@]} -eq 1 && "${processed_targets[0]}" =~ ^[A-Za-z]:[\\/].* ]]; then
  # Single absolute path: pass as positional argument (no --pyargs)
  pyargs_section="${processed_targets[0]}"
else
  if [[ ${#processed_targets[@]} -gt 0 ]]; then
    # join with spaces for --pyargs
    joined_targets="${processed_targets[*]}"
    pyargs_section="--pyargs $joined_targets"
  fi
fi

pytest_command_args="$options $pyargs_section"

# Debug/log final resolved invocation pieces.
echo "Resolved parallel marker: $marker_for_parallel_tests"
echo "Resolved sequential marker: $marker_for_sequential_tests"
echo "Final pytest command args: $pytest_command_args"

# Run tests in parallel.
echo "Running parallel tests with: pytest -m \"$marker_for_parallel_tests\" $pytest_command_args"
pytest -v -rs -o junit_suite_name=${envname} \
  --junitxml=pytest_${envname}.xml -m "$marker_for_parallel_tests" -n 6 --import-mode=importlib ${pytest_args} ${pytest_command_args}
status1=$?

# Run tests sequentially.
echo "Running sequential tests with: pytest -m \"$marker_for_sequential_tests\" $pytest_command_args"
pytest -v -rs -o junit_suite_name=${envname}_no_xdist \
  --junitxml=pytest_${envname}_no_xdist.xml -m "$marker_for_sequential_tests" --import-mode=importlib ${pytest_args} ${pytest_command_args}
status2=$?

# Exit logic: fail only if both were "no tests selected" or one had a true error.
if [[ $status1 == 5 && $status2 == 5 ]]; then
  exit $status1
fi
if [[ $status1 != 0 && $status1 != 5 ]]; then
  exit $status1
fi
if [[ $status2 != 0 && $status2 != 5 ]]; then
  exit $status2
fi
import pytest
import os
import unittest

# Add this decorator to force the test to run in the main process
@pytest.mark.no_xdist
class TestEnvironmentSpy(unittest.TestCase):
    def test_what_is_my_environment(self):
        """
        This test prints the environment variable and then intentionally fails
        so we can see the output in the logs.
        """
        timeout_val = os.environ.get('TC_TIMEOUT')
        print(f"\n--- SPY TEST: Inside pytest, TC_TIMEOUT is: {timeout_val} ---\n")
        
        self.fail("This spy test intentionally fails to report the environment.")
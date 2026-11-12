"""Unit tests for the retry decision policy."""

import unittest
from llm.retry import evaluate_retry


class TestRetryPolicy(unittest.TestCase):

    def test_never_retry_after_first_token(self):
        """Never retry if failure occurred after first token arrived."""
        dec = evaluate_retry(status_code=500, retry_count=0, first_token_received=True)
        self.assertFalse(dec.should_retry)
        self.assertIn("first token", dec.reason.lower())

    def test_client_errors_never_retry(self):
        """Never retry permanent 4xx errors (400, 401, 403)."""
        for code in (400, 401, 403):
            dec = evaluate_retry(status_code=code, retry_count=0, first_token_received=False)
            self.assertFalse(dec.should_retry)
            self.assertEqual(dec.sleep_s, 0.0)

    def test_rate_limit_retry_with_header(self):
        """429 respects retry-after header when provided."""
        dec = evaluate_retry(
            status_code=429,
            retry_count=0,
            first_token_received=False,
            retry_after_header=2.5,
            max_retries=2,
        )
        self.assertTrue(dec.should_retry)
        self.assertGreaterEqual(dec.sleep_s, 2.5)

    def test_rate_limit_exceeding_max_retries(self):
        """429 stops retrying when max_retries is reached."""
        dec = evaluate_retry(
            status_code=429,
            retry_count=2,
            first_token_received=False,
            max_retries=2,
        )
        self.assertFalse(dec.should_retry)

    def test_server_error_retry_once(self):
        """5xx allows exactly one retry before first token."""
        dec1 = evaluate_retry(status_code=503, retry_count=0, first_token_received=False)
        self.assertTrue(dec1.should_retry)

        dec2 = evaluate_retry(status_code=503, retry_count=1, first_token_received=False)
        self.assertFalse(dec2.should_retry)


if __name__ == "__main__":
    unittest.main()

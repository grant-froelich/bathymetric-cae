# tests/run_all_tests.py
"""
Test runner script for all test modules.
"""

import pytest
import sys
import logging
from pathlib import Path


def run_all_tests():
    """Run all tests with comprehensive reporting."""
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Ensure test directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("expert_reviews").mkdir(exist_ok=True)
    
    # Test configuration
    pytest_args = [
        "tests/",
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
        f"--junitxml=test_results.xml",
        f"--html=test_report.html",
        "--self-contained-html"
    ]
    
    # Add coverage if available
    try:
        import coverage
        pytest_args.extend([
            "--cov=config",
            "--cov=core", 
            "--cov=models",
            "--cov=processing",
            "--cov=review",
            "--cov=utils",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    except ImportError:
        logging.warning("Coverage not available - install pytest-cov for coverage reports")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_tests())
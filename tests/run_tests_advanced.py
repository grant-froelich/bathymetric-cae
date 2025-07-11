# tests/run_tests_advanced.py

"""
Advanced test runner with comprehensive reporting and analysis.
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

import pytest


class TestResults:
    """Container for test execution results."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.error_tests = 0
        self.coverage_percentage = 0.0
        self.test_durations = {}
        self.failed_test_details = []
        self.performance_warnings = []
    
    @property
    def execution_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def success_rate(self):
        if self.total_tests > 0:
            return (self.passed_tests / self.total_tests) * 100
        return 0
    
    def to_dict(self):
        return {
            'execution_time': self.execution_time,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'error_tests': self.error_tests,
            'success_rate': self.success_rate,
            'coverage_percentage': self.coverage_percentage,
            'test_durations': self.test_durations,
            'failed_test_details': self.failed_test_details,
            'performance_warnings': self.performance_warnings
        }


class AdvancedTestRunner:
    """Advanced test runner with comprehensive analysis."""
    
    def __init__(self):
        self.results = TestResults()
        self.config = self._load_test_config()
        self.logger = self._setup_logging()
    
    def _load_test_config(self) -> Dict:
        """Load test configuration."""
        config_file = Path("tests/test_config.json")
        
        default_config = {
            "coverage_threshold": 85.0,
            "performance_thresholds": {
                "test_execution_time": 300.0,  # 5 minutes
                "individual_test_time": 30.0,   # 30 seconds
                "memory_usage_mb": 2000
            },
            "test_categories": {
                "unit": {"pattern": "test_*.py", "exclude": ["test_integration.py", "test_performance.py"]},
                "integration": {"pattern": "test_integration.py"},
                "performance": {"pattern": "test_performance.py"}
            },
            "reporting": {
                "generate_html": True,
                "generate_junit": True,
                "generate_coverage": True,
                "send_notifications": False
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load test config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup test runner logging."""
        logger = logging.getLogger('test_runner')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/test_runner.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_tests(self, test_category: str = "all", verbose: bool = True, 
                  parallel: bool = False, coverage: bool = True) -> TestResults:
        """Run tests with specified configuration."""
        self.logger.info(f"Starting test execution - Category: {test_category}")
        self.results.start_time = time.time()
        
        try:
            # Prepare test environment
            self._prepare_test_environment()
            
            # Build pytest arguments
            pytest_args = self._build_pytest_args(test_category, verbose, parallel, coverage)
            
            self.logger.info(f"Executing pytest with args: {' '.join(pytest_args)}")
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            # Process results
            self._process_test_results(exit_code)
            
            # Generate reports
            self._generate_reports()
            
            # Performance analysis
            self._analyze_performance()
            
            # Check quality gates
            self._check_quality_gates()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            raise
        finally:
            self.results.end_time = time.time()
            self.logger.info(f"Test execution completed in {self.results.execution_time:.2f} seconds")
        
        return self.results
    
    def _prepare_test_environment(self):
        """Prepare test environment."""
        # Create required directories
        directories = ["logs", "plots", "expert_reviews", "test_data", "htmlcov"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        # Generate test data if needed
        if not Path("test_data").glob("*.bag"):
            self.logger.info("Generating test data...")
            try:
                from tests.test_fixtures.sample_data_generator import TestDataGenerator
                TestDataGenerator.create_test_dataset(Path("test_data"), 5)
            except Exception as e:
                self.logger.warning(f"Could not generate test data: {e}")
        
        # Check system requirements
        self._check_system_requirements()
    
    def _check_system_requirements(self):
        """Check system requirements for testing."""
        requirements = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            requirements.append("Python 3.8+ required")
        
        # Check available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            if available_memory < 2:
                requirements.append(f"Low available memory: {available_memory:.1f}GB")
        except ImportError:
            pass
        
        # Check GDAL availability
        try:
            from osgeo import gdal
        except ImportError:
            requirements.append("GDAL not available - some tests may be skipped")
        
        # Check TensorFlow
        try:
            import tensorflow as tf
            if len(tf.config.list_physical_devices('GPU')) == 0:
                self.logger.info("No GPU detected - using CPU for tests")
        except ImportError:
            requirements.append("TensorFlow not available")
        
        if requirements:
            self.logger.warning("System requirement issues:")
            for req in requirements:
                self.logger.warning(f"  - {req}")
    
    def _build_pytest_args(self, test_category: str, verbose: bool, 
                          parallel: bool, coverage: bool) -> List[str]:
        """Build pytest command arguments."""
        args = ["tests/"]
        
        # Verbosity
        if verbose:
            args.append("-v")
        
        # Test selection
        if test_category != "all":
            category_config = self.config["test_categories"].get(test_category, {})
            pattern = category_config.get("pattern")
            exclude = category_config.get("exclude", [])
            
            if pattern:
                args = [f"tests/{pattern}"]
            
            for exclude_pattern in exclude:
                args.extend(["--ignore", f"tests/{exclude_pattern}"])
        
        # Parallel execution
        if parallel:
            try:
                import pytest_xdist
                args.extend(["-n", "auto"])
            except ImportError:
                self.logger.warning("pytest-xdist not available - running sequentially")
        
        # Coverage
        if coverage:
            coverage_args = [
                "--cov=config", "--cov=core", "--cov=models", 
                "--cov=processing", "--cov=review", "--cov=utils",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-report=term-missing"
            ]
            args.extend(coverage_args)
        
        # Reporting
        if self.config["reporting"]["generate_junit"]:
            args.extend(["--junitxml=test-results.xml"])
        
        if self.config["reporting"]["generate_html"]:
            args.extend(["--html=test-report.html", "--self-contained-html"])
        
        # Performance and quality
        args.extend([
            "--tb=short",
            "--strict-markers",
            "--durations=10",
            f"--maxfail=10"
        ])
        
        return args
    
    def _process_test_results(self, exit_code: int):
        """Process test execution results."""
        # Parse JUnit XML if available
        junit_file = Path("test-results.xml")
        if junit_file.exists():
            self._parse_junit_results(junit_file)
        
        # Parse coverage results
        coverage_file = Path("coverage.xml")
        if coverage_file.exists():
            self._parse_coverage_results(coverage_file)
        
        # Determine overall success
        self.results.exit_code = exit_code
    
    def _parse_junit_results(self, junit_file: Path):
        """Parse JUnit XML results."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # Extract test statistics
            testsuite = root.find('.//testsuite')
            if testsuite is not None:
                self.results.total_tests = int(testsuite.get('tests', 0))
                self.results.failed_tests = int(testsuite.get('failures', 0))
                self.results.error_tests = int(testsuite.get('errors', 0))
                self.results.skipped_tests = int(testsuite.get('skipped', 0))
                self.results.passed_tests = (self.results.total_tests - 
                                           self.results.failed_tests - 
                                           self.results.error_tests - 
                                           self.results.skipped_tests)
            
            # Extract test durations and failures
            for testcase in root.findall('.//testcase'):
                name = testcase.get('name', 'unknown')
                classname = testcase.get('classname', '')
                time_str = testcase.get('time', '0')
                
                try:
                    duration = float(time_str)
                    full_name = f"{classname}::{name}" if classname else name
                    self.results.test_durations[full_name] = duration
                    
                    # Check for slow tests
                    threshold = self.config["performance_thresholds"]["individual_test_time"]
                    if duration > threshold:
                        self.results.performance_warnings.append({
                            'test': full_name,
                            'duration': duration,
                            'threshold': threshold,
                            'type': 'slow_test'
                        })
                except ValueError:
                    pass
                
                # Extract failure details
                failure = testcase.find('failure')
                error = testcase.find('error')
                if failure is not None or error is not None:
                    failure_info = failure if failure is not None else error
                    self.results.failed_test_details.append({
                        'test': f"{classname}::{name}",
                        'type': 'failure' if failure is not None else 'error',
                        'message': failure_info.get('message', ''),
                        'details': failure_info.text or ''
                    })
                    
        except Exception as e:
            self.logger.error(f"Error parsing JUnit results: {e}")
    
    def _parse_coverage_results(self, coverage_file: Path):
        """Parse coverage XML results."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Extract overall coverage
            coverage_elem = root.find('.//coverage')
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get('line-rate', 0))
                self.results.coverage_percentage = line_rate * 100
                
        except Exception as e:
            self.logger.error(f"Error parsing coverage results: {e}")
    
    def _generate_reports(self):
        """Generate comprehensive test reports."""
        self.logger.info("Generating test reports...")
        
        # Generate summary report
        self._generate_summary_report()
        
        # Generate performance report
        self._generate_performance_report()
        
        # Generate failure analysis
        if self.results.failed_test_details:
            self._generate_failure_analysis()
    
    def _generate_summary_report(self):
        """Generate test execution summary report."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_summary': {
                'total_execution_time': self.results.execution_time,
                'total_tests': self.results.total_tests,
                'passed_tests': self.results.passed_tests,
                'failed_tests': self.results.failed_tests,
                'skipped_tests': self.results.skipped_tests,
                'error_tests': self.results.error_tests,
                'success_rate': self.results.success_rate
            },
            'quality_metrics': {
                'coverage_percentage': self.results.coverage_percentage,
                'coverage_threshold': self.config['coverage_threshold'],
                'coverage_passed': self.results.coverage_percentage >= self.config['coverage_threshold']
            },
            'performance_summary': {
                'total_warnings': len(self.results.performance_warnings),
                'slow_tests': len([w for w in self.results.performance_warnings if w['type'] == 'slow_test']),
                'execution_time_threshold': self.config['performance_thresholds']['test_execution_time'],
                'execution_time_passed': self.results.execution_time <= self.config['performance_thresholds']['test_execution_time']
            }
        }
        
        # Save summary report
        with open('test-summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Test summary: {self.results.passed_tests}/{self.results.total_tests} passed "
                        f"({self.results.success_rate:.1f}%), Coverage: {self.results.coverage_percentage:.1f}%")
    
    def _generate_performance_report(self):
        """Generate performance analysis report."""
        if not self.results.test_durations:
            return
        
        # Sort tests by duration
        sorted_durations = sorted(self.results.test_durations.items(), 
                                key=lambda x: x[1], reverse=True)
        
        performance_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_execution_time': self.results.execution_time,
            'slowest_tests': sorted_durations[:10],  # Top 10 slowest
            'performance_warnings': self.results.performance_warnings,
            'statistics': {
                'total_tests_timed': len(self.results.test_durations),
                'average_test_time': sum(self.results.test_durations.values()) / len(self.results.test_durations),
                'median_test_time': sorted(self.results.test_durations.values())[len(self.results.test_durations)//2],
                'max_test_time': max(self.results.test_durations.values()),
                'min_test_time': min(self.results.test_durations.values())
            }
        }
        
        with open('test-performance.json', 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        if self.results.performance_warnings:
            self.logger.warning(f"Performance warnings: {len(self.results.performance_warnings)}")
            for warning in self.results.performance_warnings[:5]:  # Show first 5
                self.logger.warning(f"  {warning['test']}: {warning['duration']:.2f}s > {warning['threshold']}s")
    
    def _generate_failure_analysis(self):
        """Generate failure analysis report."""
        failure_analysis = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_failures': len(self.results.failed_test_details),
            'failure_categories': {},
            'detailed_failures': self.results.failed_test_details
        }
        
        # Categorize failures
        for failure in self.results.failed_test_details:
            category = self._categorize_failure(failure)
            if category not in failure_analysis['failure_categories']:
                failure_analysis['failure_categories'][category] = 0
            failure_analysis['failure_categories'][category] += 1
        
        with open('test-failures.json', 'w') as f:
            json.dump(failure_analysis, f, indent=2)
        
        self.logger.error(f"Test failures: {len(self.results.failed_test_details)}")
        for category, count in failure_analysis['failure_categories'].items():
            self.logger.error(f"  {category}: {count}")
    
    def _categorize_failure(self, failure: Dict) -> str:
        """Categorize test failure by type."""
        message = failure.get('message', '').lower()
        details = failure.get('details', '').lower()
        
        if 'assertion' in message or 'assert' in details:
            return 'assertion_error'
        elif 'import' in message or 'modulenotfound' in message:
            return 'import_error'
        elif 'timeout' in message or 'timeout' in details:
            return 'timeout_error'
        elif 'memory' in message or 'memory' in details:
            return 'memory_error'
        elif 'gdal' in message or 'gdal' in details:
            return 'gdal_error'
        elif 'tensorflow' in message or 'tf' in details:
            return 'tensorflow_error'
        else:
            return 'other_error'
    
    def _analyze_performance(self):
        """Analyze test performance."""
        # Check overall execution time
        threshold = self.config['performance_thresholds']['test_execution_time']
        if self.results.execution_time > threshold:
            self.results.performance_warnings.append({
                'type': 'overall_execution_time',
                'duration': self.results.execution_time,
                'threshold': threshold,
                'message': f"Total execution time {self.results.execution_time:.2f}s exceeds threshold {threshold}s"
            })
        
        # Analyze memory usage if available
        try:
            import psutil
            current_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            memory_threshold = self.config['performance_thresholds']['memory_usage_mb']
            
            if current_memory > memory_threshold:
                self.results.performance_warnings.append({
                    'type': 'memory_usage',
                    'memory_mb': current_memory,
                    'threshold': memory_threshold,
                    'message': f"Memory usage {current_memory:.1f}MB exceeds threshold {memory_threshold}MB"
                })
        except ImportError:
            pass
    
    def _check_quality_gates(self):
        """Check quality gates and determine if tests pass quality requirements."""
        quality_issues = []
        
        # Coverage gate
        if self.results.coverage_percentage < self.config['coverage_threshold']:
            quality_issues.append(
                f"Coverage {self.results.coverage_percentage:.1f}% below threshold {self.config['coverage_threshold']}%"
            )
        
        # Failure rate gate
        if self.results.success_rate < 95.0:  # 95% success rate required
            quality_issues.append(
                f"Success rate {self.results.success_rate:.1f}% below 95%"
            )
        
        # Performance gate
        if len(self.results.performance_warnings) > 5:
            quality_issues.append(
                f"Too many performance warnings: {len(self.results.performance_warnings)}"
            )
        
        if quality_issues:
            self.logger.error("Quality gate failures:")
            for issue in quality_issues:
                self.logger.error(f"  - {issue}")
            return False
        else:
            self.logger.info("All quality gates passed")
            return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Advanced Test Runner for Enhanced Bathymetric CAE")
    
    parser.add_argument(
        '--category', 
        choices=['all', 'unit', 'integration', 'performance'], 
        default='all',
        help='Test category to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Skip coverage analysis'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (unit tests only, no coverage)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    args = parser.parse_args()
    
    # Adjust settings for quick run
    if args.quick:
        args.category = 'unit'
        args.no_coverage = True
    
    # Special handling for benchmark mode
    if args.benchmark:
        args.category = 'performance'
        args.parallel = False  # Performance tests should run sequentially
    
    try:
        runner = AdvancedTestRunner()
        results = runner.run_tests(
            test_category=args.category,
            verbose=args.verbose,
            parallel=args.parallel,
            coverage=not args.no_coverage
        )
        
        # Print final summary
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Tests: {results.total_tests}")
        print(f"Passed: {results.passed_tests}")
        print(f"Failed: {results.failed_tests}")
        print(f"Skipped: {results.skipped_tests}")
        print(f"Errors: {results.error_tests}")
        print(f"Success Rate: {results.success_rate:.1f}%")
        print(f"Coverage: {results.coverage_percentage:.1f}%")
        print(f"Execution Time: {results.execution_time:.2f}s")
        
        if results.performance_warnings:
            print(f"Performance Warnings: {len(results.performance_warnings)}")
        
        print("="*60)
        
        # Exit with appropriate code
        if results.failed_tests > 0 or results.error_tests > 0:
            print("❌ Tests FAILED")
            return 1
        elif results.coverage_percentage < runner.config['coverage_threshold']:
            print("⚠️  Tests PASSED but coverage below threshold")
            return 1
        else:
            print("✅ All tests PASSED")
            return 0
            
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
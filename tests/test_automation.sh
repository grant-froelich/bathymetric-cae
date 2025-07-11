#!/bin/bash

# test_automation.sh
# Comprehensive test automation script for Enhanced Bathymetric CAE Processing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON=${PYTHON:-python3}
TEST_DIR="tests"
LOG_DIR="logs"
REPORT_DIR="test_reports"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo -e "${NC}"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python version
    python_version=$($PYTHON --version 2>&1 | cut -d' ' -f2)
    required_version="3.8"
    
    if ! $PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ required, found $python_version"
        exit 1
    fi
    
    # Check required packages
    required_packages=("pytest" "numpy" "tensorflow")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! $PYTHON -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_error "Missing required packages: ${missing_packages[*]}"
        log_info "Install with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    # Check optional packages
    optional_packages=("pytest-cov" "pytest-html" "pytest-xdist")
    for package in "${optional_packages[@]}"; do
        package_module=$(echo "$package" | sed 's/-/_/g')
        if ! $PYTHON -c "import $package_module" 2>/dev/null; then
            log_warning "Optional package $package not found"
        fi
    done
    
    log_success "All dependencies satisfied"
}

setup_environment() {
    log_info "Setting up test environment..."
    
    # Create directories
    mkdir -p "$LOG_DIR" "$REPORT_DIR" "test_data" "plots" "expert_reviews" "htmlcov"
    
    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    
    # Generate test data if needed
    if [ ! -f "test_data/dataset_metadata.json" ]; then
        log_info "Generating test data..."
        $PYTHON -c "
from tests.test_fixtures.sample_data_generator import TestDataGenerator
from pathlib import Path
try:
    TestDataGenerator.create_test_dataset(Path('test_data'), 5)
    print('Test data generated successfully')
except Exception as e:
    print(f'Warning: Could not generate test data - {e}')
" || log_warning "Could not generate test data"
    fi
    
    log_success "Environment setup complete"
}

run_lint_checks() {
    log_info "Running code quality checks..."
    
    # Check if linting tools are available
    if command -v flake8 >/dev/null 2>&1; then
        log_info "Running flake8..."
        flake8 tests/ --max-line-length=100 --extend-ignore=E203,W503 || log_warning "Flake8 issues found"
    fi
    
    if command -v black >/dev/null 2>&1; then
        log_info "Checking code formatting with black..."
        black --check tests/ --line-length=100 || log_warning "Code formatting issues found"
    fi
    
    if command -v isort >/dev/null 2>&1; then
        log_info "Checking import sorting..."
        isort --check-only tests/ || log_warning "Import sorting issues found"
    fi
}

run_security_checks() {
    log_info "Running security checks..."
    
    if command -v bandit >/dev/null 2>&1; then
        log_info "Running bandit security scan..."
        bandit -r . -f json -o "$REPORT_DIR/bandit_report.json" -x tests/ || log_warning "Security issues found"
    fi
    
    if command -v safety >/dev/null 2>&1; then
        log_info "Checking for known security vulnerabilities..."
        safety check --json --output "$REPORT_DIR/safety_report.json" || log_warning "Vulnerable packages found"
    fi
}

run_unit_tests() {
    print_header "Running Unit Tests"
    
    $PYTHON -m pytest tests/test_*.py \
        --ignore=tests/test_integration.py \
        --ignore=tests/test_performance.py \
        -v \
        --tb=short \
        --strict-markers \
        --junitxml="$REPORT_DIR/unit_test_results.xml" \
        --html="$REPORT_DIR/unit_test_report.html" \
        --self-contained-html \
        --cov=config --cov=core --cov=models --cov=processing --cov=review --cov=utils \
        --cov-report=html:htmlcov \
        --cov-report=xml:"$REPORT_DIR/coverage.xml" \
        --cov-report=term-missing \
        || return 1
    
    log_success "Unit tests completed"
}

run_integration_tests() {
    print_header "Running Integration Tests"
    
    $PYTHON -m pytest tests/test_integration.py \
        -v \
        --tb=short \
        --strict-markers \
        --junitxml="$REPORT_DIR/integration_test_results.xml" \
        --html="$REPORT_DIR/integration_test_report.html" \
        --self-contained-html \
        || return 1
    
    log_success "Integration tests completed"
}

run_performance_tests() {
    print_header "Running Performance Tests"
    
    $PYTHON -m pytest tests/test_performance.py \
        -v \
        --tb=short \
        --strict-markers \
        --junitxml="$REPORT_DIR/performance_test_results.xml" \
        --html="$REPORT_DIR/performance_test_report.html" \
        --self-contained-html \
        || return 1
    
    log_success "Performance tests completed"
}

run_advanced_tests() {
    print_header "Running Advanced Test Suite"
    
    $PYTHON tests/run_tests_advanced.py \
        --category all \
        --verbose \
        || return 1
    
    log_success "Advanced test suite completed"
}

generate_reports() {
    log_info "Generating comprehensive reports..."
    
    # Move reports to report directory
    [ -f "test-summary.json" ] && mv test-summary.json "$REPORT_DIR/"
    [ -f "test-performance.json" ] && mv test-performance.json "$REPORT_DIR/"
    [ -f "test-failures.json" ] && mv test-failures.json "$REPORT_DIR/"
    [ -f "coverage.xml" ] && mv coverage.xml "$REPORT_DIR/"
    
    # Generate consolidated report
    cat > "$REPORT_DIR/test_execution_summary.md" << EOF
# Test Execution Summary

**Timestamp:** $(date)
**Environment:** $(uname -s) $(uname -r)
**Python Version:** $($PYTHON --version)

## Test Results

### Unit Tests
- Results: $REPORT_DIR/unit_test_results.xml
- Report: $REPORT_DIR/unit_test_report.html

### Integration Tests  
- Results: $REPORT_DIR/integration_test_results.xml
- Report: $REPORT_DIR/integration_test_report.html

### Performance Tests
- Results: $REPORT_DIR/performance_test_results.xml
- Report: $REPORT_DIR/performance_test_report.html

## Coverage
- HTML Report: htmlcov/index.html
- XML Report: $REPORT_DIR/coverage.xml

## Quality Checks
- Security: $REPORT_DIR/bandit_report.json
- Dependencies: $REPORT_DIR/safety_report.json

## Logs
- Test Runner: $LOG_DIR/test_runner.log
- Processing: $LOG_DIR/bathymetric_processing.log
EOF
    
    log_success "Reports generated in $REPORT_DIR/"
}

cleanup() {
    log_info "Cleaning up test artifacts..."
    
    # Remove temporary files but keep important reports
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clean up test data if specified
    if [ "$CLEANUP_TEST_DATA" = "true" ]; then
        rm -rf test_data/
        log_info "Test data cleaned up"
    fi
    
    log_success "Cleanup completed"
}

show_help() {
    cat << EOF
Enhanced Bathymetric CAE Test Automation Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    unit            Run unit tests only
    integration     Run integration tests only
    performance     Run performance tests only
    all             Run all test suites (default)
    advanced        Run advanced test suite with comprehensive analysis
    lint            Run code quality checks only
    security        Run security checks only
    setup           Setup test environment only
    clean           Clean up test artifacts

Options:
    -h, --help      Show this help message
    -v, --verbose   Verbose output
    -f, --fail-fast Exit on first test failure
    -p, --parallel  Run tests in parallel (where supported)
    -c, --coverage  Generate coverage reports
    -q, --quick     Quick test run (unit tests only, no coverage)
    --cleanup       Clean up test data after tests
    --no-deps       Skip dependency checks

Examples:
    $0                          # Run all tests
    $0 unit                     # Run unit tests only
    $0 --quick                  # Quick unit test run
    $0 advanced --verbose       # Full test suite with detailed output
    $0 lint security           # Run quality and security checks only

EOF
}

# Parse command line arguments
COMMANDS=()
VERBOSE=false
FAIL_FAST=false
PARALLEL=false
COVERAGE=true
QUICK=false
CLEANUP_TEST_DATA=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        --cleanup)
            CLEANUP_TEST_DATA=true
            shift
            ;;
        --no-deps)
            SKIP_DEPS=true
            shift
            ;;
        unit|integration|performance|all|advanced|lint|security|setup|clean)
            COMMANDS+=("$1")
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to 'all' if no commands specified
if [ ${#COMMANDS[@]} -eq 0 ]; then
    COMMANDS=("all")
fi

# Quick mode adjustments
if [ "$QUICK" = true ]; then
    COMMANDS=("unit")
    COVERAGE=false
    log_info "Quick mode: Running unit tests only, no coverage"
fi

# Main execution
main() {
    print_header "Enhanced Bathymetric CAE Test Automation"
    
    # Setup
    if [ "$SKIP_DEPS" != true ]; then
        check_dependencies
    fi
    
    # Execute commands
    for command in "${COMMANDS[@]}"; do
        case $command in
            setup)
                setup_environment
                ;;
            lint)
                run_lint_checks
                ;;
            security)
                run_security_checks
                ;;
            unit)
                setup_environment
                run_unit_tests
                ;;
            integration)
                setup_environment
                run_integration_tests
                ;;
            performance)
                setup_environment
                run_performance_tests
                ;;
            advanced)
                setup_environment
                run_advanced_tests
                ;;
            all)
                setup_environment
                run_lint_checks
                run_security_checks
                
                # Run tests with error handling
                test_results=0
                
                run_unit_tests || test_results=$?
                if [ $test_results -ne 0 ] && [ "$FAIL_FAST" = true ]; then
                    log_error "Unit tests failed, stopping (fail-fast mode)"
                    exit $test_results
                fi
                
                run_integration_tests || test_results=$?
                if [ $test_results -ne 0 ] && [ "$FAIL_FAST" = true ]; then
                    log_error "Integration tests failed, stopping (fail-fast mode)"
                    exit $test_results
                fi
                
                run_performance_tests || test_results=$?
                if [ $test_results -ne 0 ] && [ "$FAIL_FAST" = true ]; then
                    log_error "Performance tests failed, stopping (fail-fast mode)"
                    exit $test_results
                fi
                
                # Generate final reports
                generate_reports
                
                if [ $test_results -ne 0 ]; then
                    log_error "Some tests failed"
                    exit $test_results
                fi
                ;;
            clean)
                cleanup
                ;;
            *)
                log_error "Unknown command: $command"
                exit 1
                ;;
        esac
    done
    
    # Cleanup if requested
    if [ "$CLEANUP_TEST_DATA" = true ]; then
        cleanup
    fi
    
    log_success "Test automation completed successfully!"
}

# Trap to ensure cleanup on exit
trap 'log_info "Script interrupted, cleaning up..."; cleanup; exit 1' INT TERM

# Run main function
main

exit 0
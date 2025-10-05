#!/usr/bin/env python3
"""
Test runner script for Kepler Exoplanet Detection test suite

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --unit             # Run only unit tests
    python tests/run_tests.py --integration      # Run only integration tests
    python tests/run_tests.py --performance      # Run only performance tests
    python tests/run_tests.py --coverage         # Run with coverage report
    python tests/run_tests.py --html             # Generate HTML report
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run pytest with specified options"""

    # Base pytest command
    cmd = ['pytest']

    # Add verbosity
    cmd.append('-v')

    # Add markers based on arguments
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    elif args.performance:
        cmd.extend(['-m', 'performance'])
    elif args.compatibility:
        cmd.extend(['-m', 'compatibility'])

    # Add coverage options
    if args.coverage:
        cmd.extend([
            '--cov=.',
            '--cov-report=term-missing',
            '--cov-report=html',
            '--cov-fail-under=80'
        ])

    # Add HTML report
    if args.html:
        cmd.extend([
            '--html=tests/reports/test_report.html',
            '--self-contained-html'
        ])

    # Add parallel execution
    if args.parallel:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        cmd.extend(['-n', str(num_cores)])

    # Add specific test file or directory
    if args.path:
        cmd.append(args.path)
    else:
        cmd.append('tests/')

    # Add additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args)

    # Print command
    print(f"Running: {' '.join(cmd)}\n")

    # Create reports directory
    reports_dir = Path('tests/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Run pytest
    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run Kepler Exoplanet Detection test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    test_group.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    test_group.add_argument(
        '--performance',
        action='store_true',
        help='Run only performance tests'
    )
    test_group.add_argument(
        '--compatibility',
        action='store_true',
        help='Run only compatibility tests'
    )

    # Additional options
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML test report'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Specific test file or directory to run'
    )
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments to pass to pytest'
    )

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(args)

    # Print summary
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()

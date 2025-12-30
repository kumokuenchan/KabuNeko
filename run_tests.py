#!/usr/bin/env python3
"""
Test Runner Script for Stock Analysis Toolkit

This script provides an easy way to run all tests with various options.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Extra verbose output
"""

import sys
import subprocess
import argparse


def run_tests(args):
    """Run pytest with specified arguments"""

    # Base pytest command
    cmd = ['python', '-m', 'pytest']

    # Add test directory
    cmd.append('tests/')

    # Verbosity
    if args.verbose:
        cmd.append('-vv')
    else:
        cmd.append('-v')

    # Test selection
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])

    # Skip slow tests
    if args.fast:
        cmd.extend(['-m', 'not slow'])

    # Coverage
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])

    # Specific test file
    if args.file:
        cmd = ['python', '-m', 'pytest', f'tests/{args.file}', '-v']

    # Run specific test
    if args.test:
        cmd.extend(['-k', args.test])

    # Show output
    if args.show_output:
        cmd.append('-s')

    # Stop on first failure
    if args.fail_fast:
        cmd.append('-x')

    print("=" * 70)
    print("Running Stock Analysis Toolkit Tests")
    print("=" * 70)
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run tests
    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run tests for Stock Analysis Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Run all tests
  %(prog)s --unit              Run unit tests only
  %(prog)s --fast              Skip slow tests
  %(prog)s --coverage          Generate coverage report
  %(prog)s --file test_data.py Run specific test file
  %(prog)s --test sma          Run tests matching 'sma'
  %(prog)s -x -s               Stop on first failure, show output
        """
    )

    parser.add_argument(
        '--unit', action='store_true',
        help='Run only unit tests'
    )

    parser.add_argument(
        '--integration', action='store_true',
        help='Run only integration tests'
    )

    parser.add_argument(
        '--fast', action='store_true',
        help='Skip slow tests'
    )

    parser.add_argument(
        '--coverage', action='store_true',
        help='Generate coverage report'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Extra verbose output'
    )

    parser.add_argument(
        '--file', type=str,
        help='Run specific test file (e.g., test_data.py)'
    )

    parser.add_argument(
        '--test', '-k', type=str,
        help='Run tests matching pattern'
    )

    parser.add_argument(
        '--show-output', '-s', action='store_true',
        help='Show print statements and output'
    )

    parser.add_argument(
        '--fail-fast', '-x', action='store_true',
        help='Stop on first test failure'
    )

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(args)

    # Print summary
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()

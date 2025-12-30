#!/usr/bin/env python3
"""
Notebook Verification Script

This script verifies that all Jupyter notebooks are valid and can be loaded.
It checks for syntax errors and structural issues.

Usage:
    python verify_notebooks.py
"""

import os
import json
import sys
from pathlib import Path


def verify_notebook(notebook_path):
    """Verify a single notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Check basic structure
        assert 'cells' in notebook, "Missing 'cells' key"
        assert 'metadata' in notebook, "Missing 'metadata' key"
        assert 'nbformat' in notebook, "Missing 'nbformat' key"

        # Count cells
        code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
        markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')

        return {
            'valid': True,
            'total_cells': len(notebook['cells']),
            'code_cells': code_cells,
            'markdown_cells': markdown_cells,
            'error': None
        }

    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'error': f"JSON decode error: {str(e)}"
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f"Error: {str(e)}"
        }


def main():
    """Main verification function"""
    print("=" * 70)
    print("Jupyter Notebook Verification")
    print("=" * 70)

    # Find all notebooks
    notebooks_dir = Path('notebooks')
    notebooks = list(notebooks_dir.glob('**/*.ipynb'))

    if not notebooks:
        print("\n❌ No notebooks found!")
        return 1

    print(f"\nFound {len(notebooks)} notebook(s) to verify\n")

    all_valid = True
    results = []

    for notebook_path in sorted(notebooks):
        relative_path = notebook_path.relative_to('notebooks')
        print(f"Checking: {relative_path}...", end=' ')

        result = verify_notebook(notebook_path)
        result['path'] = str(relative_path)
        results.append(result)

        if result['valid']:
            print(f"✅ Valid ({result['total_cells']} cells: "
                  f"{result['code_cells']} code, {result['markdown_cells']} markdown)")
        else:
            print(f"❌ Invalid")
            print(f"   Error: {result['error']}")
            all_valid = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    valid_count = sum(1 for r in results if r['valid'])
    print(f"\nTotal notebooks: {len(results)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {len(results) - valid_count}")

    if all_valid:
        print("\n✅ All notebooks are valid!")
        return 0
    else:
        print("\n❌ Some notebooks have errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())

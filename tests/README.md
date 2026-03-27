# Test Suite

This directory contains tests for the JNJ project module structure and imports.

## Test Scripts

### 1. `test_imports_structural.py`

**Purpose:** Verify the module structure without external dependencies

**What it checks:**
- All required `__init__.py` files exist
- All Python files have valid syntax
- Module hierarchy is correct
- Import statements are present and syntactically valid

**Usage:**
```bash
python3 test_imports_structural.py
```

**Requirements:** None (uses only Python standard library)

**Output:**
```
✓ All structural checks passed!
✓ Module structure is correct
✓ All Python files have valid syntax
✓ All __init__.py files are properly formatted
```

---

### 2. `test_imports_with_mocks.py`

**Purpose:** Test actual module imports using mocks for heavy dependencies

**What it checks:**
- All modules can be imported successfully
- Key classes are properly exported
- No import errors due to module structure issues

**Usage:**
```bash
python3 test_imports_with_mocks.py
```

**Requirements:** None (uses mocks for torch, transformers, etc.)

**Output:**
```
Module imports:     16/27 passed (others fail due to missing external packages)
Class imports:      4/6 passed (others fail due to missing external packages)
✓ All models imports successful!
✓ All losses imports successful!
```

---

## Understanding Test Results

### Expected Failures

The following failures are **expected and OK**:
- `data` modules fail due to missing `h5py`
- `evaluation` modules fail due to missing `sklearn`
- `utils` modules fail due to missing `sklearn`

These are external runtime dependencies that are not needed for structural verification.

### Actual Failures

Any other failures indicate structural issues that need fixing:
- Syntax errors in Python files
- Missing `__init__.py` files
- Incorrect relative imports
- Missing module exports

---

## Running All Tests

```bash
# Run structural tests first (fast, no dependencies)
python3 test_imports_structural.py

# Run import tests with mocks
python3 test_imports_with_mocks.py
```

Both should show **all structural checks passed**.

---

## Module Structure

The project is organized into 5 main modules:

```
models/      - All deep learning model architectures
data/        - Data loading, preprocessing, padding, augmentation
losses/      - Loss functions and training utilities
evaluation/  - Metrics, post-processing, visualization
utils/       - Common utilities and helper functions
```

Each module has:
- `__init__.py` - Exports public API
- Implementation files (e.g., `components.py`, `loading.py`, etc.)

---

## Fixing Issues

If tests fail, check:

1. **Syntax errors:** Run structural test to identify file
2. **Missing files:** Verify all files exist in their directories
3. **Import errors:** Check that all relative imports use dot notation (`.module`)
4. **Circular imports:** Verify module dependencies form a DAG (no cycles)

---

## Integration with CI/CD

These tests are suitable for:
- Pre-commit hooks (use `test_imports_structural.py`)
- CI/CD pipelines (use `test_imports_structural.py` for fast feedback)
- Local development (run both tests before committing)

Example GitHub Actions:
```yaml
- name: Check module structure
  run: python3 tests/test_imports_structural.py

- name: Test imports with mocks
  run: python3 tests/test_imports_with_mocks.py
```

---

## Historical Records

See `IMPORT_TEST_REPORT.md` for:
- Detailed test results
- Issues found and fixed
- Module structure documentation
- Dependency information

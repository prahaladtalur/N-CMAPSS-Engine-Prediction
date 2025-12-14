# N-CMAPSS Project Cleanup Report

This document outlines the cleanup and refactoring recommendations for the N-CMAPSS Engine Prediction project.

## Executive Summary

The codebase has several areas that need cleanup:
- **Dead code**: Redundant LSTM model file, unused functions
- **Dependency inconsistencies**: Mismatch between requirements.txt and pyproject.toml
- **Code organization**: Duplicate visualization scripts, redundant wrapper files
- **Documentation gaps**: Missing docstrings, unused imports
- **Consistency issues**: Inconsistent naming and file organization

---

## 1. Code Organization Issues

### 1.1 Redundant Files

#### `src/models/lstm_model.py` - **RECOMMENDED FOR REMOVAL**
- **Issue**: This file contains `build_lstm_model()` which duplicates functionality already in `architectures.py` as `LSTMModel`
- **Impact**: The LSTM model is already available through the model registry system
- **Action**: Remove this file and update imports
- **Files affected**: 
  - `src/models/__init__.py` (remove `build_lstm_model` import)
  - Any code importing from `lstm_model.py`

#### `visualize_all.py` - **CONSIDER MERGING**
- **Issue**: Duplicates functionality in `scripts/example_visualizations.py`
- **Impact**: Two scripts doing similar things creates confusion
- **Action**: Either merge into `scripts/example_visualizations.py` or remove if redundant
- **Recommendation**: Keep `scripts/example_visualizations.py` as it's more comprehensive

#### `main.py` - **SIMPLIFY OR REMOVE**
- **Issue**: Just a thin wrapper around `train_model.py` that adds `.env` loading
- **Impact**: Adds unnecessary indirection
- **Action**: Either:
  - Option A: Remove `main.py` and add `.env` loading to `train_model.py`
  - Option B: Keep but document it's just a convenience wrapper
- **Recommendation**: Keep but simplify, or merge functionality into `train_model.py`

### 1.2 Directory Structure Recommendations

Current structure is mostly good, but consider:
- Moving `visualize_all.py` to `scripts/` if kept
- Ensuring all root-level scripts have clear purposes

---

## 2. Dead Code Removal

### 2.1 Unused Functions

#### `build_lstm_model()` in `src/models/lstm_model.py`
- **Status**: Not used anywhere (replaced by `LSTMModel` in `architectures.py`)
- **Action**: Remove entire file

#### `get_model_summary()` in `src/models/lstm_model.py`
- **Status**: Not used anywhere
- **Action**: Remove with file

#### `train_lstm()` in `src/models/train.py`
- **Status**: Marked as "legacy" but still exported
- **Action**: 
  - Option A: Remove if no external dependencies
  - Option B: Keep but mark as deprecated with clear migration path
- **Recommendation**: Remove after verifying no external usage

#### `prepare_sequences()` export
- **Status**: Exported from `src/models/__init__.py` but likely only used internally
- **Action**: Make internal (remove from `__init__.py`) if not needed externally

### 2.2 Unused Imports

#### `scipy.stats` in `src/utils/visualize.py`
- **Status**: Imported but never used
- **Action**: Remove import

#### `Callable` in `src/models/architectures.py`
- **Status**: Imported but not used
- **Action**: Remove from imports

### 2.3 Unused Files

- `src/models/lstm_model.py` - Entire file can be removed (see 1.1)

---

## 3. Dependency Issues

### 3.1 Inconsistencies Between `requirements.txt` and `pyproject.toml`

#### Missing in `pyproject.toml` but in `requirements.txt`:
- `torch>=2.0.0` - **NOT USED** (project uses TensorFlow)
- `pytorch-lightning>=2.0.0` - **NOT USED** (project uses TensorFlow)
- `scipy>=1.9.0` - **PARTIALLY USED** (imported but `stats` not used)

#### Missing in `requirements.txt` but in `pyproject.toml`:
- `pandas` - Used by `rul-datasets` dependency
- `wandb>=0.15.0` - Used for experiment tracking
- `python-dotenv>=1.0.0` - Used in `main.py`

#### Recommendations:
1. **Remove unused dependencies**:
   - Remove `torch` and `pytorch-lightning` from `requirements.txt` (project uses TensorFlow)
   - Remove `scipy` if only `stats` was needed (currently unused)

2. **Synchronize dependencies**:
   - Use `pyproject.toml` as the single source of truth
   - Remove `requirements.txt` OR keep it synchronized
   - **Recommendation**: Remove `requirements.txt` and use only `pyproject.toml` (project already uses `uv`)

3. **Add missing dependencies to `requirements.txt`** (if keeping it):
   - `wandb>=0.15.0`
   - `python-dotenv>=1.0.0`
   - `pandas` (if explicitly needed)

---

## 4. Documentation Gaps

### 4.1 Missing Docstrings

All major functions have docstrings - **GOOD** ✅

### 4.2 Missing Comments

- Complex logic in `architectures.py` (TCN, WaveNet blocks) could use more inline comments
- Attention mechanism implementation could use more explanation

### 4.3 Public API Documentation

- `src/models/__init__.py` exports both legacy and new APIs - should document migration path
- `src/utils/__init__.py` has good exports but could use module-level docstring

---

## 5. Consistency Issues

### 5.1 Naming Conventions

- **GOOD**: Most functions use snake_case ✅
- **GOOD**: Classes use PascalCase ✅
- **ISSUE**: Some inconsistency in file naming (e.g., `train_model.py` vs `lstm_model.py`)

### 5.2 Code Style

- **GOOD**: Project uses Black formatter (configured in `pyproject.toml`) ✅
- **ISSUE**: Some files have shebangs (`#!/usr/bin/env python`), some don't
- **Recommendation**: Standardize - add shebangs to all executable scripts

### 5.3 Import Organization

- **GOOD**: Imports are generally well-organized ✅
- **ISSUE**: Some files import unused modules (see 2.2)

---

## 6. Configuration Cleanup

### 6.1 `pyproject.toml`

- **GOOD**: Well-structured ✅
- **ISSUE**: MyPy config has `python_version = "3.9"` but project requires `>=3.8`
- **Action**: Update to match minimum version or clarify Python version support

### 6.2 `Makefile`

- **GOOD**: Comprehensive and well-documented ✅
- **MINOR**: Some targets reference files that may be removed (e.g., if `main.py` is removed)

---

## 7. Recommended Action Plan

### Priority 1 (High Impact, Low Risk)
1. ✅ Remove unused `scipy.stats` import
2. ✅ Remove unused `Callable` import
3. ✅ Remove `torch` and `pytorch-lightning` from `requirements.txt`
4. ✅ Update `src/models/__init__.py` to remove dead code exports

### Priority 2 (Medium Impact, Medium Risk)
1. ⚠️ Remove or deprecate `src/models/lstm_model.py`
2. ⚠️ Remove or simplify `main.py`
3. ⚠️ Remove `visualize_all.py` or merge into `scripts/example_visualizations.py`
4. ⚠️ Make `prepare_sequences()` internal (remove from exports)

### Priority 3 (Low Impact, Low Risk)
1. 📝 Add module-level docstrings
2. 📝 Add inline comments to complex logic
3. 📝 Standardize shebangs
4. 📝 Update MyPy Python version config

---

## 8. Files to Modify

### Files to Delete:
- `src/models/lstm_model.py` (after verification)

### Files to Modify:
- `src/models/__init__.py` - Remove dead code exports
- `src/models/train.py` - Remove or deprecate `train_lstm()`
- `src/utils/visualize.py` - Remove unused `scipy.stats` import
- `src/models/architectures.py` - Remove unused `Callable` import
- `requirements.txt` - Remove unused dependencies or remove file entirely
- `pyproject.toml` - Update MyPy config if needed
- `main.py` - Simplify or merge into `train_model.py`
- `visualize_all.py` - Remove or move to `scripts/`

---

## 9. Testing Recommendations

After cleanup:
1. Run all existing tests (if any)
2. Verify model training still works
3. Verify visualization scripts still work
4. Check that imports don't break

---

## 10. Summary Statistics

- **Files to delete**: 1-3 files
- **Files to modify**: 6-8 files
- **Dead code lines**: ~150-200 lines
- **Unused dependencies**: 2-3 packages
- **Estimated cleanup time**: 2-4 hours

---

---

## 11. Cleanup Actions Completed ✅

### Completed Changes:

1. ✅ **Removed unused imports**:
   - Removed `scipy.stats` from `src/utils/visualize.py`
   - Removed `Callable` from `src/models/architectures.py`

2. ✅ **Cleaned up dead code**:
   - Removed `src/models/lstm_model.py` (redundant with `LSTMModel` in `architectures.py`)
   - Updated `src/models/__init__.py` to remove dead code exports
   - Added deprecation notice to `train_lstm()` function

3. ✅ **Fixed dependency inconsistencies**:
   - Removed `torch` and `pytorch-lightning` from `requirements.txt` (project uses TensorFlow)
   - Added missing dependencies (`wandb`, `python-dotenv`, `tensorflow`) to `requirements.txt`
   - Added note in `requirements.txt` that `pyproject.toml` is the primary source

4. ✅ **Improved documentation**:
   - Added module-level docstrings to `src/utils/__init__.py` and `src/data/__init__.py`
   - Added detailed comments to complex code (AttentionLayer, TCNBlock)
   - Improved documentation for `visualize_all.py`

5. ✅ **Code organization**:
   - Deleted `visualize_all.py` (redundant with `scripts/example_visualizations.py`)
   - Deleted `main.py` (merged `.env` loading into `train_model.py`)
   - Updated `Makefile` and `README.md` to remove references to deleted files

### Files Modified:
- ✅ `src/utils/visualize.py` - Removed unused `scipy.stats` import
- ✅ `src/models/architectures.py` - Removed unused `Callable` import, added detailed comments
- ✅ `src/models/__init__.py` - Cleaned up exports, removed dead code references
- ✅ `src/models/train.py` - Added deprecation notice to `train_lstm()`
- ✅ `src/utils/__init__.py` - Added comprehensive module docstring
- ✅ `src/data/__init__.py` - Added module docstring
- ✅ `requirements.txt` - Fixed dependencies (removed torch/pytorch-lightning, added missing ones)
- ✅ `train_model.py` - Added `.env` file loading (merged from `main.py`)
- ✅ `Makefile` - Updated to use `train_model.py` instead of `main.py`
- ✅ `README.md` - Removed references to deleted files

### Files Deleted:
- ✅ `src/models/lstm_model.py` - Redundant file removed (LSTM available via model registry)
- ✅ `main.py` - Removed redundant wrapper (functionality merged into `train_model.py`)
- ✅ `visualize_all.py` - Removed redundant script (use `scripts/example_visualizations.py` instead)

---

*Report generated: 2025-01-XX*
*Last updated: 2025-01-XX*
*Cleanup completed: 2025-01-XX*


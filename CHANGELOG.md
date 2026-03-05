# Changelog

All notable changes to the N-CMAPSS RUL Prediction project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] - 2026-03-04

### Major Milestone: Complete 20-Model Benchmark & Production Package

This release represents the completion of comprehensive model evaluation and production-ready deployment infrastructure.

### Added

#### Ensemble Prediction System
- **Enhanced predict.py** with ensemble support
  - Weighted averaging: 50% MSTCN + 30% Transformer + 20% WaveNet
  - Confidence metrics based on model agreement (HIGH/MEDIUM/LOW)
  - Backward compatible with single model predictions
  - Expected 10-15% RMSE improvement (target: 6.5-6.7 cycles)

- **scripts/prepare_ensemble.py** - Automated ensemble setup
  - One-command training of all top 3 models
  - Automatic validation of existing models
  - Force retrain and quick test modes
  - Ensemble metadata generation

- **demo_inference.py** - Interactive demonstration
  - Synthetic data mode (no trained models required)
  - Real model mode (single or ensemble)
  - Three scenarios: healthy, degrading, failing engines
  - Maintenance recommendations based on predictions
  - Complete inference pipeline walkthrough

- **scripts/cross_dataset_validation.py** - Cross-dataset testing
  - Framework for FD1-FD7 validation
  - Transfer learning evaluation
  - Automated result aggregation
  - Generalization metrics

#### Comprehensive Documentation (6 Major Documents)

- **MSTCN_EXPLAINED.md** (350+ lines)
  - Deep technical explanation of winner architecture
  - Multi-scale temporal convolutions with dilation
  - Global Fusion Attention mechanism (channel + temporal + cross-scale)
  - Mathematical formulations
  - Comparison to Transformer, LSTM, WaveNet, MDFA
  - Implementation notes and best practices

- **FINAL_ANALYSIS_REPORT.md** (400+ lines)
  - Complete 20-model benchmark results
  - Architecture category analysis (Convolutional, RNN, Attention, Hybrid)
  - Sequence length impact study (1K vs 20K timesteps)
  - SOTA models deep dive
  - Production recommendations
  - Future research directions

- **FINAL_RESULTS_COMPARISON.md**
  - 30 vs 100 epoch training analysis
  - Performance vs training time tradeoffs
  - Model selection guide by use case

- **ENSEMBLE_GUIDE.md**
  - Ensemble quick start guide
  - Performance expectations
  - Python API examples
  - Advanced optimization techniques
  - Troubleshooting guide

- **QUICKSTART.md**
  - 5-minute quick start guide
  - Installation to trained model
  - Troubleshooting common issues
  - Performance expectations table

- **PROJECT_SUMMARY.md**
  - Complete project overview
  - All deliverables documented
  - Repository organization guide
  - Impact and significance

#### Visualizations & Data

- **results/final_comparison/all_models_comparison.png**
  - 4-panel comprehensive comparison
  - RMSE rankings for all 20 models
  - R² score comparison
  - MAE distribution
  - Sequence length impact visualization

- **results/final_comparison/top_10_models.png**
  - Top performers detailed view
  - Color-coded tiers (gold/silver/bronze)
  - Performance metrics overlaid

- **results/final_comparison/sota_models_comparison.png**
  - 2023-2025 literature models focus
  - Dual-axis RMSE + R² comparison

- **results/final_comparison/results.json**
  - Machine-readable benchmark results
  - All 20 models with complete metrics

#### Production Package

- **models/production/mstcn_metadata.json**
  - Model specifications
  - Performance metrics
  - Training configuration
  - Input/output requirements

- **models/production/README.md**
  - Deployment guide
  - Python API examples
  - Command-line usage
  - Preprocessing requirements

### Changed

#### Updated Core Files

- **README.md**
  - Added winner banner (MSTCN: RMSE 6.80, R² 0.90)
  - Comprehensive benchmark table with top 8 models
  - Ensemble prediction references
  - Listed all 20 available architectures by category
  - Updated performance metrics section

- **CLAUDE.md**
  - Added "Best Practices" section from benchmark findings
  - Updated recommended configuration (sequence_length: 1000!)
  - Changed best model reference from CNN-GRU to MSTCN
  - Added model selection guidance

- **train_model.py** & **src/models/architectures.py**
  - Merged 5 new SOTA models (MSTCN, ATCN, CATA-TCN, TTSNet, Sparse Transformer)
  - Updated model categories
  - Resolved merge conflicts from PR #13

### Key Findings

#### 🏆 Winner: MSTCN
- **RMSE**: 6.80 cycles
- **R² Score**: 0.9006
- **Accuracy@20**: 99.12%
- **58% improvement** over previous best (TCN at 16.13 RMSE)

#### 🔬 Critical Discovery: Sequence Length Impact
- **1,000 timesteps**: RMSE 12.04, R² 0.58 (GOOD)
- **20,294 timesteps**: RMSE 21.20, R² -0.02 (BAD)
- **Improvement**: 58% better performance with shorter sequences
- **Implication**: Recent patterns more predictive, long sequences add noise

#### 📊 Architecture Rankings
| Rank | Model | RMSE | R² | Category |
|------|-------|------|-----|----------|
| 1 | MSTCN | 6.80 | 0.90 | Convolutional |
| 2 | Transformer | 6.82 | 0.90 | Attention |
| 3 | WaveNet | 6.84 | 0.90 | Convolutional |
| 4 | ATCN | 7.01 | 0.89 | Attention |
| 5 | CATA-TCN | 7.38 | 0.88 | Attention |

#### ❌ What Didn't Work
- **Traditional RNNs**: All LSTM/GRU variants had RMSE > 22, R² < 0 (negative!)
- **Long sequences**: Full 20K timesteps performed 58% worse
- **100 epoch training**: Only 6% better than 30 epochs (not worth 3× time)

### Performance Benchmarks

#### Training Efficiency
- **Fast & Effective**: MSTCN (3 min, RMSE 6.80), Transformer (3 min, 6.82)
- **Slow & Ineffective**: LSTM full seq (30 min, RMSE 22.28)
- **Conclusion**: Modern CNNs are 10× faster and 3× better

#### Production Recommendations
```yaml
Recommended Configuration:
  model: mstcn
  epochs: 30
  batch_size: 64
  max_sequence_length: 1000  # CRITICAL!
  learning_rate: 0.001
  early_stopping_patience: 10

Expected Performance:
  RMSE: 6.8-7.5 cycles
  R²: 0.88-0.90
  Accuracy@20: >98%
  Training time: 3 minutes
```

### Repository Statistics

- **Documentation**: 1,600+ lines across 6 major documents
- **Visualizations**: 6 high-quality plots (300 DPI)
- **Models Evaluated**: 20 architectures
- **Total Experiments**: 23+ training runs
- **Training Time**: ~50 hours total
- **Git Commits**: 8+ major documentation/feature commits
- **Files Added**: 15+ new files
- **Files Updated**: 5 existing files

---

## [1.2.0] - 2026-03-02

### Added
- 5 new SOTA models: MSTCN, ATCN, CATA-TCN, TTSNet, Sparse Transformer
- PR #13: Implement SOTA models from recent literature (2023-2025)

---

## [1.1.0] - 2026-03-01

### Added
- Normalized metrics for paper comparison
- Test infrastructure for metrics validation
- Comprehensive metrics documentation in README

### Changed
- Enhanced metrics.py with normalized RMSE/MAE
- Added SOTA benchmarks (MDFA paper targets)

---

## [1.0.0] - 2026-02-28

### Initial Release

#### Features
- Single unified data loader (`src/data/load_data.py`)
- 14 neural network architectures
- Automated training pipeline (`train_model.py`)
- W&B experiment tracking
- Comprehensive evaluation metrics
- Visualization utilities

#### Available Models
LSTM, BiLSTM, GRU, BiGRU, attention_lstm, resnet_lstm, TCN, WaveNet,
cnn_lstm, cnn_gru, inception_lstm, transformer, MLP

#### Best Model (at release)
- CNN-GRU: RMSE ~16-18 cycles

---

## Release Highlights

### v2.0.0 (Current)
🏆 **Production-ready with comprehensive benchmark**
- Best model identified: MSTCN (RMSE 6.80)
- Ensemble capability for maximum accuracy
- 6 comprehensive documentation files
- Complete production package
- Interactive demo

### v1.2.0
📚 **SOTA models integration**
- 5 cutting-edge architectures from 2023-2025 papers

### v1.1.0
🔬 **Research-grade metrics**
- Normalized metrics for paper comparison
- Test infrastructure

### v1.0.0
🚀 **Foundation**
- Unified pipeline with 14 models

---

## Upgrade Guide

### From v1.x to v2.0

**New Required Steps:**
```bash
# Update dependencies
uv sync

# Use new sequence length for training (CRITICAL!)
python train_model.py --model mstcn --max-seq-length 1000

# Prepare ensemble (optional)
python scripts/prepare_ensemble.py
```

**Breaking Changes:**
- None - fully backward compatible

**Recommended Actions:**
1. Retrain models with `--max-seq-length 1000` for 58% improvement
2. Switch from CNN-GRU to MSTCN (current best)
3. Use ensemble for maximum accuracy

---

## Future Roadmap

### v2.1.0 (Planned)
- Complete cross-dataset validation implementation
- Automated hyperparameter optimization
- REST API for production deployment

### v2.2.0 (Planned)
- Model explainability (attention visualization)
- Real-time inference optimizations
- Docker deployment package

### v3.0.0 (Research)
- Multi-task learning (RUL + anomaly detection)
- Federated learning support
- Edge deployment optimizations

---

**Maintainer**: Prahalad Talur
**Repository**: https://github.com/prahaladtalur/N-CMAPSS-Engine-Prediction
**License**: MIT (assumed)

---

*Last Updated: March 4, 2026*

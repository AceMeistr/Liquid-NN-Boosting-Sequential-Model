# CHIEF ARCHITECT DEVELOPMENT PLAN — ModelMK1

## Liquid Neural Network + XGBoost Hybrid | Institutional-Grade Evolution Roadmap

**Classification:** Internal — Model Development Division  
**Version:** 2.0  
**Date:** 2026-03-12  
**Scope:** Full-spectrum enhancement plan covering Accuracy, Robustness, Efficiency, Effectiveness, Fail-Safes, Overfit/Underfit Prevention, and Institutional Rigor  

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Audit](#2-current-architecture-audit)
3. [Phase 1 — Data Pipeline Hardening](#3-phase-1--data-pipeline-hardening)
4. [Phase 2 — Feature Engineering Evolution](#4-phase-2--feature-engineering-evolution)
5. [Phase 3 — Model Architecture Advancements](#5-phase-3--model-architecture-advancements)
6. [Phase 4 — Training Regime Overhaul](#6-phase-4--training-regime-overhaul)
7. [Phase 5 — Overfitting & Underfitting Prevention Framework](#7-phase-5--overfitting--underfitting-prevention-framework)
8. [Phase 6 — Hyperparameter Optimization Expansion](#8-phase-6--hyperparameter-optimization-expansion)
9. [Phase 7 — Evaluation & Backtesting Rigor](#9-phase-7--evaluation--backtesting-rigor)
10. [Phase 8 — Fail-Safe & Resilience Engineering](#10-phase-8--fail-safe--resilience-engineering)
11. [Phase 9 — Inference & Production Hardening](#11-phase-9--inference--production-hardening)
12. [Phase 10 — Institutional Infrastructure](#12-phase-10--institutional-infrastructure)
13. [Phase 11 — Advanced Techniques & Research Frontier](#13-phase-11--advanced-techniques--research-frontier)
14. [Retail-Algorithmic Model Limitation Matrix](#14-retail-algorithmic-model-limitation-matrix)
15. [Implementation Priority Matrix](#15-implementation-priority-matrix)
16. [Appendix: Technique Reference](#16-appendix-technique-reference)
17. [Inference Speed Analysis — 1-Minute and 5-Minute Bars](#17-inference-speed-analysis--1-minute-and-5-minute-bars)
18. [Aggressive Optuna Early Pruning Strategy](#18-aggressive-optuna-early-pruning-strategy)
19. [Public Data Sources, Institutional Knowledge & Architectural Advantages](#19-public-data-sources-institutional-knowledge--architectural-advantages)

---

## 1. EXECUTIVE SUMMARY

ModelMK1 is a hybrid ensemble combining a Liquid Neural Network (MarketLNN: Temporal Attention + LTC/GRU backbone + Squeeze-Excite gating) with XGBoost residual boosting for financial time-series range prediction. The current system follows a Build → Tune → Train → Hybrid → Predict → Backtest pipeline operating on 355,944 samples (954 parquet files, 2022–2026).

**Current Baseline Metrics:**
- Best Optuna val_loss: ~77.48
- Directional accuracy: ~49.95%
- Architecture: 44-feature input → 64-step sequences → hidden_size 128 → scalar regression
- Tuning: 30-trial Optuna (TPE + MedianPruner)

**Gap Analysis:** The model exhibits characteristics common to retail-algorithmic projects — fixed feature engineering, single-step prediction, no regime awareness, naive backtesting (no slippage/commissions), no walk-forward validation, no model versioning, no drift detection, limited explainability, and a narrow hyperparameter search space. Directional accuracy near 50% indicates the model is not yet capturing meaningful signal above random.

This plan provides **142 concrete improvements** across **11 phases**, designed to transform ModelMK1 from a retail prototype into an institutional-grade prediction system.

---

## 2. CURRENT ARCHITECTURE AUDIT

### 2.1 Strengths (Retain)

| Component | Assessment |
|-----------|-----------|
| Hybrid LNN + XGBoost residual ensemble | Sound architectural choice — temporal model + tabular booster complement each other |
| Vectorized sliding-window construction (`stride_tricks`) | 50-100x faster than naive loops — keep |
| Chronological train/val split with scaler fit on train only | Correct — prevents look-ahead bias |
| HuberLoss for regression | Robust to outliers vs. MSE — retain as option |
| EMA weight averaging | Stabilizes inference — proven technique |
| Mixed precision (AMP + GradScaler) | 2-3x speedup on CUDA with minimal accuracy loss |
| Optuna TPE sampler with persistent SQLite | Good Bayesian optimizer choice with resume capability |
| ONNX export + validation | Production-ready inference format |
| Atomic file I/O with retry | Prevents corruption on crash |

### 2.2 Critical Weaknesses (Must Fix)

| ID | Weakness | Severity | Impact |
|----|----------|----------|--------|
| W-01 | Directional accuracy ~50% (random baseline) | **CRITICAL** | Model captures no learned signal |
| W-02 | Single chronological split — no walk-forward validation | **CRITICAL** | Overfitting to specific market regime undetectable |
| W-03 | Backtest uses `sign(pred) * y_true` — no slippage, fees, market impact | **HIGH** | PnL metrics are fictional; real performance will be worse |
| W-04 | Fixed hand-coded feature windows (SMA-14, RSI-14, EMA-12/26) | **HIGH** | Ignores optimal windows for this specific dataset |
| W-05 | Signed-range target confounds magnitude + direction into single scalar | **HIGH** | Model must solve two tasks simultaneously; conflicting gradients |
| W-06 | No regime detection or market state awareness | **HIGH** | Single model for all conditions; catastrophic in regime changes |
| W-07 | No data drift detection post-deployment | **HIGH** | Silent degradation in production |
| W-08 | XGBoost hyperparameters NOT tuned via Optuna (fixed defaults) | **MEDIUM** | Suboptimal residual correction |
| W-09 | No gradient checkpointing → limited model scaling on 6GB GPU | **MEDIUM** | Cannot test deeper architectures |
| W-10 | StandardScaler assumes Gaussian — financial data has fat tails | **MEDIUM** | Extreme values distort normalization |
| W-11 | No model versioning; training overwrites `lnn_best.pt` | **MEDIUM** | No rollback, no A/B comparison |
| W-12 | No attention visualization or SHAP integration | **LOW** | Black-box decisions; regulatory risk |

---

## 3. PHASE 1 — DATA PIPELINE HARDENING

### 3.1 Robust Scaling Strategy

**Problem:** `StandardScaler` assumes Gaussian distributions. Financial data exhibits fat tails, skewness, and structural breaks.

**Solution — Adaptive Scaler Selection:**

```
┌─────────────────────────────────────────────┐
│          ADAPTIVE SCALING STRATEGY          │
├─────────────────────────────────────────────┤
│ 1. Compute per-feature kurtosis & skewness  │
│ 2. If kurtosis > 5 OR |skew| > 2:          │
│    → RobustScaler (median/IQR)              │
│ 3. If feature is bounded [0,1]:             │
│    → MinMaxScaler (preserve range)          │
│ 4. If heavy-tailed + positive:              │
│    → QuantileTransformer (Gaussian output)  │
│ 5. Default:                                 │
│    → StandardScaler (z-score)               │
│ 6. ALWAYS: Winsorize at 1st/99th percentile │
│    BEFORE scaling to bound extreme outliers  │
└─────────────────────────────────────────────┘
```

**Implementation Details:**
- Add `WinsorizedRobustScaler` wrapper class that clips at configurable percentiles before scaling
- Store per-feature scaler type in the scaler artifact (joblib) for reproducible inference
- Add `QuantileTransformer(output_distribution='normal')` as option for features with extreme kurtosis (e.g., volume, OBV)

### 3.2 Target Engineering Overhaul

**Problem:** Current signed-range target (`range * direction`) conflates two distinct signals — magnitude and direction — into a single scalar. This forces the regression head to learn both tasks simultaneously, creating opposing gradient pressures near the zero-crossing boundary where small directional errors flip sign and cause large target errors.

**Solution — Decouple targets into a multi-task formulation:**

```
OLD:  target = signed_range = (max - min) * sign(mean_future - price)
      → Single scalar; ambiguous near zero

NEW:  target_direction  = sign(mean(future) - price)     → {-1, 0, +1}
      target_magnitude  = log1p(max - min)                → Positive scalar, log-compressed
      target_volatility = std(future_returns)              → Regime indicator
```

**Multi-Task Head Design:**

```
MarketLNN backbone → shared_representation (B, hidden_size)
                     ├─→ direction_head  → BCEWithLogitsLoss (binary: up/down)
                     ├─→ magnitude_head  → HuberLoss (positive regression)
                     └─→ volatility_head → HuberLoss (regime awareness)

Combined loss = α * L_direction + β * L_magnitude + γ * L_volatility
  where α, β, γ are learnable or scheduled weights (GradNorm or uncertainty weighting)
```

**Advantages:**
- Direction head can specialize on sign prediction (directly optimizes directional accuracy)
- Magnitude head learns scale without sign confusion
- Volatility head gives the model regime awareness
- Multi-task regularization prevents overfitting by forcing shared representations to be more general

### 3.3 Dynamic Resampling

**Problem:** Fixed 1-minute bars may not be optimal. Higher timeframes reduce noise, lower timeframes preserve granularity.

**Solution — Multi-Resolution Input:**

```
Pipeline Option A: User-configurable resample frequency
  --resample-freq 1min|5min|15min|1h|4h

Pipeline Option B: Multi-Resolution Fusion (advanced)
  Input: [1-min features (T=64), 5-min features (T=64), 15-min features (T=64)]
  Each resolution → separate LNN encoder → concatenate → shared prediction head
```

### 3.4 Data Quality Firewall

Add validation gates at each pipeline stage:

```
GATE 1 — Raw Input Validation:
  ✓ Minimum price > 0
  ✓ high >= low (no inverted bars)
  ✓ volume >= 0
  ✓ Timestamps monotonically increasing
  ✓ No gaps > 5x median interval (flag as market close)
  ✓ Price change per bar < 10% (reject fat-finger outliers)

GATE 2 — Feature Validation (post-indicators):
  ✓ No NaN/Inf in any feature column
  ✓ RSI in [0, 100], BB_pct_b in [-0.5, 1.5] (reasonable bounds)
  ✓ Feature correlation check: drop features with |corr| > 0.98 (multicollinearity)
  ✓ Feature variance check: drop near-constant features (var < 1e-8)

GATE 3 — Supervised Dataset Validation:
  ✓ Target mean ≈ 0 (approximately balanced direction)
  ✓ Target std within [1, 1000] (not degenerate)
  ✓ Sequence count > 10,000 (minimum for meaningful training)
  ✓ Train/val target distribution KS-test p > 0.01 (not wildly different)
```

### 3.5 Feature Caching Layer

**Problem:** Indicators recomputed from scratch for each Optuna trial. With 30 trials × 355K rows × 44 features, this wastes hours of computation.

**Solution:**

```python
# Cache indexed by (data_hash, resample_freq, include_stoch_rsi, seq_len, horizon)
CACHE_DIR = outputs/model/.feature_cache/

def get_or_compute_features(df, freq, include_stoch_rsi, seq_len, horizon):
    cache_key = hashlib.sha256(
        f"{df.shape}_{df.iloc[0].timestamp}_{df.iloc[-1].timestamp}"
        f"_{freq}_{include_stoch_rsi}_{seq_len}_{horizon}"
    ).hexdigest()[:16]
    
    cache_path = CACHE_DIR / f"{cache_key}.npz"
    if cache_path.exists():
        return load_cached(cache_path)
    
    bundle = build_supervised_data(df, seq_len, horizon, include_stoch_rsi)
    save_cached(cache_path, bundle)
    return bundle
```

**Impact:** Reduces Optuna 30-trial runtime from ~30× data pipeline cost to ~1× data pipeline cost + 30× training cost. On 355K rows this saves approximately 25-28 redundant feature computations.

---

## 4. PHASE 2 — FEATURE ENGINEERING EVOLUTION

### 4.1 Advanced Temporal Feature Extraction (Beyond Basic Conv1D)

**Problem:** Indicator windows are hard-coded (SMA-14, RSI-14, EMA-12/26). These are arbitrary defaults from traditional technical analysis, not optimized for this specific dataset or prediction task. A naive Conv1D stem (3 layers, fixed kernels) is marginally better — it learns one scale per layer but misses multi-scale temporal interactions.

**Superior Alternatives Ranked by Effectiveness:**

#### Option A: InceptionTime (RECOMMENDED — Best CNN for Time Series)

InceptionTime (Fawaz et al., 2020) is the state-of-the-art CNN architecture for time series classification/regression, winning most UCR benchmark competitions.

```
Raw OHLCV (B, T, 5)
    ↓
[Inception Module × 3 stacked]:
  Each module applies PARALLEL convolutions at multiple scales:
  ┌─ Conv1D(C_in, C_out/4, kernel=1)   ← point-wise (captures instantaneous)
  ├─ Conv1D(C_in, C_out/4, kernel=3)   ← short-term (3-bar patterns)
  ├─ Conv1D(C_in, C_out/4, kernel=7)   ← medium-term (7-bar patterns)
  ├─ Conv1D(C_in, C_out/4, kernel=15)  ← long-term (15-bar patterns)
  └─ MaxPool1D(3) → Conv1D(C_in, C_out/4, kernel=1)  ← max-pool branch
  All branches → Concatenate → BatchNorm → GELU → Residual
    ↓
Multi-scale features (B, T, C_out)  where C_out = 64
    ↓
[Concatenate with hand-coded indicators]  (B, T, 64+44=108)
    ↓
[Linear projection to hidden_size]
```

**Why InceptionTime beats basic Conv1D:**
- Captures patterns at 4+ temporal scales simultaneously (1, 3, 7, 15 bars)
- MaxPool branch captures local extrema (equivalent to learned support/resistance)
- Residual connections enable deeper stacking without degradation
- Only ~30% more parameters than single-scale Conv1D for 4× the pattern diversity
- Proven on 128 time-series benchmarks — consistently top-2 performer

#### Option B: TimesNet (Wu et al., 2023 — ICLR)

Transforms 1D time series into 2D space to capture intra-period and inter-period patterns:

```
Input (B, T, C)
    ↓
[FFT] → Identify top-k dominant periods (e.g., 5-min, 15-min, 1-hour cycles)
    ↓
For each period p:
  Reshape: (B, T, C) → (B, T/p, p, C)  ← 2D representation
  Apply 2D Conv (InceptionBlock2D) → capture both intra-period and inter-period
  Reshape back to (B, T, C)
    ↓
[Adaptive aggregation of all period representations]
    ↓
Output (B, T, C)  with multi-period awareness
```

**Why TimesNet is powerful for financial data:**
- Automatically discovers dominant market cycles via FFT (no manual window selection)
- Intra-period: captures patterns within a 15-min cycle (opening range breakout)
- Inter-period: captures patterns across 15-min cycles (trend continuation)
- Self-discovers the optimal window sizes that basic Conv1D would need manually tuned

#### Option C: SCINet (Structural Causal Interleaving Network)

Recursive downsampling + interleaving for multi-resolution:

```
Input (B, T, C)
    ↓
[Split into odd/even subsequences]
  Even: x[0], x[2], x[4]...  │  Odd: x[1], x[3], x[5]...
    ↓                              ↓
  Conv1D block                 Conv1D block
    ↓                              ↓
[Interactive learning: cross-exchange information]
    ↓
[Concatenate + Residual]
    ↓
[Repeat recursively at lower resolution]
```

**Recommended Implementation:**

```
Primary: InceptionTime (proven, efficient, trivial to implement)
Advanced: TimesNet (if FFT analysis reveals strong cyclical components in data)
Fallback: Standard Conv1D stem if neither library available

All options: CONCATENATE with hand-coded indicators, never replace.
Domain indicators (RSI, MACD, Bollinger) are free information.
```

### 4.2 Temporal Feature Augmentation

Add time-aware features the current system lacks:

```
Time-of-Day Features:
  - sin(2π * minute / 1440), cos(2π * minute / 1440)  → cyclical time encoding
  - Categorical: session_type ∈ {pre_market, open, mid_day, power_hour, close, after_hours}

Calendar Features:
  - Day of week (one-hot or cyclical sin/cos)
  - Month (cyclical sin/cos)
  - Is expiry day, is FOMC day, is NFP day (event flags)
  - Days to next options expiry

Cross-Sectional Features (if multi-asset):
  - Rolling correlation with market index (SPY)
  - Relative strength vs. sector
  - Beta (rolling 20-day)
```

### 4.3 Microstructure Features

For tick/minute-level data, add market microstructure signals:

```
Orderflow Proxies:
  - Kyle's Lambda: price_impact = |Δprice| / volume (per bar)
  - Volume Imbalance: (buy_vol - sell_vol) / total_vol
    (approximate via: close > open → buy; close < open → sell)
  - Amihud Illiquidity: |return| / dollar_volume

Volatility Regime:
  - Realized Volatility (5-min, 15-min, 1-hour rolling)
  - Garman-Klass volatility estimator
  - Yang-Zhang volatility estimator (handles overnight gaps)
  - Volatility of Volatility (vol-of-vol): rolling std of Parkinson volatility

Information Flow:
  - Volume-Synchronized Probability of Informed Trading (VPIN) approximation
  - Tick rule: consecutive same-direction tick count
```

### 4.4 Feature Selection Pipeline

Replace ad-hoc feature construction with a principled pipeline:

```
1. GENERATE candidate features (100+)
   ├─ Hand-coded indicators (44 current)
   ├─ Microstructure features (~10)
   ├─ Temporal features (~8)
   └─ CNN-learned features (~64 if stem added)

2. FILTER: Remove low-variance and high-correlation features
   ├─ Variance threshold: drop if σ² < 1e-8
   ├─ Pairwise correlation: if |ρ| > 0.95, drop the one with lower univariate MI
   └─ Check for information leakage: no feature should correlate > 0.5 with future target

3. RANK: Mutual Information with target
   ├─ sklearn.feature_selection.mutual_info_regression(X, y)
   ├─ Rank all features by MI score
   └─ Top-K selection (K tunable via Optuna)

4. VALIDATE: Permutation Importance on trained model
   ├─ Shuffle each feature independently
   ├─ Measure accuracy drop → importance
   └─ Drop features with importance < 0 (hurting model)

5. FINAL: Feature importance stability check
   ├─ Run ranking across 5 temporal folds
   ├─ Keep features that rank in top-K consistently (≥4/5 folds)
   └─ Reject features that are important in only 1-2 folds (regime-specific noise)
```

### 4.5 Feature Stationarity Enforcement

**Problem:** Raw price, OBV, and cumulative features are non-stationary. Models trained on level values break when price regime shifts.

**Solution:**

```
Non-Stationary → Stationary Transforms:
  - Price  → Log returns:     log(price_t / price_{t-1})
  - Volume → Log volume:      log1p(volume)
  - OBV    → OBV returns:     diff(OBV) / (abs(OBV) + 1)
  - VWAP   → VWAP deviation:  (price - VWAP) / VWAP
  - SMA/EMA → Price ratio:    price / SMA (already done — good)

Augmented Dickey-Fuller test on each feature during build:
  - If ADF p-value > 0.05: feature is non-stationary → apply differencing
  - Log and report which features required transformation
```

---

## 5. PHASE 3 — MODEL ARCHITECTURE ADVANCEMENTS

### 5.1 Enhanced MarketLNN Backbone

#### 5.1.1 Pre-LayerNorm Transformer Blocks

Current implementation uses post-norm residuals. Switch to **Pre-LayerNorm** (GPT-2 style) for more stable deep training:

```
Current (post-norm):  x → Attention(x) + x → LayerNorm
Proposed (pre-norm):  x → LayerNorm(x) → Attention → + x

Benefits:
  - Training stability in deep networks (>4 layers)
  - Removes the need for learning rate warmup in most cases
  - Proven to converge faster on temporal tasks
```

#### 5.1.2 Rotary Positional Encoding (RoPE)

Replace sinusoidal PositionalEncoding with RoPE:

```
Current:  x = x + PE[position]  (additive, absolute position)
Proposed: Q, K = apply_rotary(Q, K, position)  (multiplicative, relative position)

Benefits:
  - Encodes RELATIVE position (distance between timesteps matters more than absolute position)
  - Extrapolates better to unseen sequence lengths
  - Zero additional parameters
  - Used by LLaMA, GPT-NeoX, and all modern transformers
```

#### 5.1.3 Gated Linear Units (GLU) in FeedForward

Replace GELU MLP with `SwiGLU` (used in LLaMA, PaLM):

```
Current:   FFN(x) = Linear₂(GELU(Linear₁(x)))
Proposed:  FFN(x) = Linear₂(SiLU(Linear_gate(x)) ⊙ Linear_up(x))

Benefits:
  - Learnable gating mechanism (multiplicative interaction)
  - Smoother gradient flow
  - Consistently outperforms GELU MLP on sequence tasks
  - ~same parameter count (three projections instead of two, but smaller hidden dim)
```

#### 5.1.4 Multi-Scale Temporal Convolution Before Attention

Add dilated causal convolutions to capture local patterns before global attention:

```
Input (B, T, C)
    ↓
[Causal Conv1D bank: dilation={1, 2, 4, 8}]  → Multi-scale local patterns
    ↓
[Concat + Linear projection]
    ↓
[Self-Attention layers]  → Global pattern mixing
    ↓
[LTC/GRU backbone]  → Sequential state tracking
```

### 5.2 Mamba State Space Model (SSM) — Replacing MoE

**Why NOT MoE:** Mixture of Experts introduces a routing bottleneck — the router network itself must learn regime boundaries from the same noisy data. With only 355K samples, expert collapse (all tokens routing to 1 expert) is near-guaranteed. MoE also multiplies parameter count by `n_experts` while only using `top-k`, wasting GPU memory on a 6GB card. Load-balancing loss adds a conflicting training objective. **MoE is an inference-time and memory bottleneck that provides marginal benefit at this data scale.**

**Solution — Mamba (Structured State Space Sequence Model):**

Mamba (Gu & Dao, 2023) is the successor to S4/S5 state space models. It provides:
- **O(n) linear-time** sequence processing vs. O(n²) for attention
- **Selective state spaces** — input-dependent state transitions that naturally adapt to market regime changes without explicit routing
- **Hardware-efficient** — parallel scan on GPU, no KV-cache, constant memory per token
- **No attention bottleneck** — processes 128-step sequences as fast as 32-step

```
Architecture — Mamba Block:

Input (B, T, C)
    ↓
[Linear expand] → (B, T, D) where D = 2*C (expansion factor)
    ↓
┌─────────────────────────────────────────────────┐
│ Branch A:                Branch B (gate):        │
│ Conv1D(D, kernel=4)      Linear(D → D)           │
│     ↓                        ↓                   │
│ SiLU activation          SiLU activation          │
│     ↓                        │                   │
│ SSM (Selective Scan):        │                   │
│   A = diag(learned)          │                   │
│   B = Linear(D → N)         │                   │
│   C = Linear(D → N)         │                   │
│   Δ = softplus(Linear(D))   │                   │
│   y = selective_scan(x,Δ,A,B,C)                  │
│     ↓                        │                   │
│ [Element-wise multiply: y ⊙ gate]                │
└─────────────────────────────────────────────────┘
    ↓
[Linear project] → (B, T, C)
    ↓
[Residual + LayerNorm]
```

**Why Mamba Handles Regimes Naturally:**
- The **Δ (delta) parameter** is input-dependent — it controls how much new information overwrites the hidden state
- In trending markets: Δ is large → state updates aggressively → tracks trend
- In mean-reverting markets: Δ is small → state is sticky → captures equilibrium
- This is **learned automatically** from data, not via an explicit router
- No load-balancing loss, no expert collapse, no routing overhead

**Proposed MarketLNN + Mamba Hybrid:**

```
Input (B, T, input_size)
    ↓
[Linear projection → hidden_size]
    ↓
[Squeeze-Excite gating]
    ↓
[Mamba Block × num_layers]  ← REPLACES attention + GRU
    ↓                         O(n) vs O(n²) attention
[LayerNorm]
    ↓
[:, -1, :]  → last timestep
    ↓
[Prediction Head]
```

**Concrete Advantages Over MoE:**

| Property | MoE | Mamba |
|----------|-----|-------|
| Complexity | O(n²) attention + routing | O(n) linear scan |
| Parameters | 4× FeedForward (4 experts) | 1× expansion (2×C) |
| GPU memory | HIGH (all expert weights loaded) | LOW (recurrent state only) |
| Regime adaptation | Explicit router (fragile) | Implicit via Δ (robust) |
| Long sequences | Quadratic slowdown | Linear — no degradation |
| 6GB GPU fit | Tight at hidden=256 | Comfortable at hidden=512 |
| Implementation | Complex (routing, balancing) | Clean (single scan kernel) |

**Dependencies:** `pip install mamba-ssm` (requires CUDA) or `pip install mamba-ssm[causal-conv1d]`

**Fallback:** If `mamba-ssm` unavailable, fall back to S4D (Structured State Space for Sequences — Diagonal), which is pure PyTorch with no custom CUDA kernels. S4D achieves ~85% of Mamba's performance with zero external dependencies.

### 5.3 Temporal Fusion Transformer (TFT) Elements

Borrow key innovations from Google's Temporal Fusion Transformer:

```
1. Variable Selection Networks:
   - Learned feature importance per timestep
   - Soft feature gating: features get multiplied by sigmoid gates
   - Different features can be important at different times

2. Static Enrichment:
   - Time-invariant features (e.g., asset class) enrich temporal representations
   - Cross-attention from temporal to static context

3. Interpretable Multi-Head Attention:
   - Attention weights are directly interpretable as "which past timesteps matter"
   - Can be visualized for explainability reports
```

### 5.4 Deeper Prediction Head

Current head: 3 linear layers (hidden → hidden/2 → hidden/4 → 1). This is shallow for a complex regression task.

**Enhanced Prediction Head:**

```
shared_representation (B, hidden)
    ↓
[Global Average Pool + Global Max Pool] → concat → (B, 2*hidden)
    ↓
Linear(2*hidden → hidden) + LayerNorm + SiLU + Dropout(0.2)
    ↓
Linear(hidden → hidden//2) + LayerNorm + SiLU + Dropout(0.1)
    ↓
Linear(hidden//2 → hidden//4) + SiLU
    ↓
├─→ Linear(hidden//4 → 1) → magnitude (ReLU, non-negative)
├─→ Linear(hidden//4 → 1) → direction (sigmoid → binary)
└─→ Linear(hidden//4 → 2) → aleatoric uncertainty (mean + log_variance)
```

### 5.5 Aleatoric + Epistemic Uncertainty Estimation

**Current:** MC Dropout only (epistemic uncertainty). Misses data-inherent noise.

**Improved — Full Bayesian Uncertainty:**

```
Epistemic (model uncertainty):
  - MC Dropout: run N forward passes with dropout enabled
  - Epistemic uncertainty = variance across N samples
  - HIGH epistemic → model is uncertain → need more training data in this regime

Aleatoric (data uncertainty):
  - Model outputs (μ, log σ²) instead of single point prediction
  - Loss = NLL: 0.5 * (log σ² + (y - μ)² / σ²)
  - HIGH aleatoric → data is inherently noisy → cannot reduce with more training
  - Crucial for financial data where some periods are genuinely unpredictable

Total Uncertainty = epistemic + aleatoric
  - Use for position sizing: small positions when total uncertainty is high
  - Use for abstention: refuse to predict when uncertainty > threshold
```

---

## 6. PHASE 4 — TRAINING REGIME OVERHAUL

### 6.1 Combinatorial Purged Cross-Validation (CPCV)

**Problem:** Single chronological 80/20 split tests on one specific market regime. Walk-Forward CV is better but still limited — it produces only k point-estimates (one per fold) with high variance, and the expanding window creates train-set size imbalance across folds.

**Solution — Combinatorial Purged Cross-Validation (CPCV) (López de Prado, 2018):**

CPCV is the gold-standard validation method for financial ML, directly estimating the **Probability of Backtest Overfitting (PBO)** while generating a full distribution of out-of-sample performance.

```
ALGORITHM:

1. PARTITION data into N equal non-overlapping groups (N=16 recommended)
   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
   │ G1 │ G2 │ G3 │ G4 │ G5 │ G6 │ G7 │ G8 │ G9 │G10│G11│G12│G13│G14│G15│G16│
   │Jan │Feb │Mar │Apr │May │Jun │Jul │Aug │Sep │Oct│Nov│Dec│Jan│Feb│Mar│Apr│
   │ 22 │ 22 │ 22 │ 22 │ 22 │ 22 │ 22 │ 22 │ 22 │ 22│ 22│ 22│ 23│ 23│ 23│ 23│
   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

2. FORM all combinations: choose k groups for testing (k = N/2 = 8)
   C(16, 8) = 12,870 unique train/test splits

3. For EACH combination:
   a. TEST set = selected k groups
   b. TRAIN set = remaining N-k groups
   c. PURGE: Remove samples from train whose target window overlaps into test
      Purge gap = 2 × horizon bars at each train/test boundary
   d. EMBARGO: Remove embargo_size bars after each test → train boundary
      Prevents serial correlation leakage
   e. Train model on purged train set → evaluate on test set
   f. Record: {combination_id, test_groups, val_loss, dir_acc, sharpe}

4. RESULT: Distribution of 12,870 out-of-sample performance values
   → Mean, std, confidence intervals — NOT a single point estimate
```

**CPCV vs Walk-Forward CV:**

| Property | Walk-Forward CV | CPCV |
|----------|----------------|------|
| # of performance estimates | 5 (one per fold) | 12,870 (all combinations) |
| Statistical power | LOW (5 samples) | HIGH (12K+ samples) |
| Train set size balance | Imbalanced (grows per fold) | Balanced (always N/2 groups) |
| Recency bias | YES (last fold has largest train) | NO (all groups equally weighted) |
| Computes PBO | NO | YES (direct metric) |
| Regime coverage | Partial (expanding window) | Complete (all permutations) |
| Computation cost | 5× training | Expensive but parallelizable |

**Probability of Backtest Overfitting (PBO):**

```
PBO CALCULATION:
  1. For each of 12,870 combinations:
     - Select the model config with best IN-SAMPLE (train) performance
     - Measure that config's OUT-OF-SAMPLE (test) rank among all configs
  2. PBO = fraction of combinations where best-IS config ranks
           BELOW MEDIAN on OOS
  3. Interpretation:
     PBO < 0.15 → LOW overfit risk → SAFE to deploy
     PBO 0.15-0.30 → MODERATE risk → deploy with caution
     PBO 0.30-0.50 → HIGH risk → needs more regularization
     PBO > 0.50 → OVERFIT → DO NOT DEPLOY (worse than random selection)
```

**Practical Implementation (Affordable Variant for 6GB GPU):**

Full CPCV with 12,870 retrains is expensive. Use the **sub-sampled CPCV** approach:

```python
import itertools
import random

N_GROUPS = 16
K_TEST = 8
MAX_COMBINATIONS = 200  # Sub-sample for tractability

all_combos = list(itertools.combinations(range(N_GROUPS), K_TEST))
sampled_combos = random.sample(all_combos, min(MAX_COMBINATIONS, len(all_combos)))

results = []
for combo in sampled_combos:
    test_groups = set(combo)
    train_groups = set(range(N_GROUPS)) - test_groups
    
    train_data = purge_and_embargo(data, train_groups, test_groups, 
                                    purge_gap=2*horizon, embargo=horizon)
    test_data = select_groups(data, test_groups)
    
    model = train_model(train_data)
    metrics = evaluate(model, test_data)
    results.append(metrics)

# Distribution of OOS performance
oos_sharpes = [r['sharpe'] for r in results]
print(f"OOS Sharpe: {mean(oos_sharpes):.3f} +/- {std(oos_sharpes):.3f}")
print(f"95% CI: [{percentile(oos_sharpes, 2.5):.3f}, {percentile(oos_sharpes, 97.5):.3f}]")
print(f"PBO estimate: {sum(1 for s in oos_sharpes if s < 0) / len(oos_sharpes):.2%}")
```

**Purge & Embargo Implementation:**

```
For each train/test boundary:

  [...TRAIN DATA...]|---PURGE---|---EMBARGO---|[...TEST DATA...]
                    ← 2×horizon → ← horizon →

  PURGE: Remove train samples whose target lookahead window
         extends into the test period. This prevents the model
         from training on information that partially overlaps
         with the test targets.
         
  EMBARGO: Remove test samples immediately after a train period
           boundary. Prevents serial autocorrelation from leaking
           train-period statistical properties into adjacent test
           samples.

  Both gaps are measured in BARS (not calendar time):
    purge_gap = 2 × horizon = 2 × 15 = 30 bars
    embargo   = 1 × horizon = 15 bars
```

**Aggregation:**
- Report **mean ± std** across all tested combinations
- If std(directional_accuracy) > 5%: model is regime-sensitive (red flag)
- Compute PBO: if > 30%, the model is likely overfit
- Final production model: train on ALL data, but only deploy if PBO < 30%

### 6.2 Curriculum Learning

**Problem:** Training on all 355K samples equally treats easy and hard samples the same. Hard samples (regime transitions, flash crashes) may overwhelm the model early in training.

**Solution — Progressive Difficulty:**

```
Stage 1 (Epochs 1-3): Easy samples only
  - Filter: |target| < median(|targets|)  → calm, predictable periods
  - Learning rate: full (1e-3)
  - Objective: learn basic temporal patterns

Stage 2 (Epochs 4-7): Medium + easy samples  
  - Filter: |target| < 75th percentile
  - Learning rate: 0.7× initial
  - Objective: generalize to moderate volatility

Stage 3 (Epochs 8+): All samples
  - No filter: full dataset including extreme moves
  - Learning rate: 0.5× initial (scheduled anyway)
  - Objective: handle all regimes including tails
```

### 6.3 Loss Function Engineering

**Problem:** HuberLoss(delta=1.0) treats all errors uniformly above delta. Financial returns have asymmetric distributions and regime-dependent characteristics.

**Solution — Composite Loss:**

```python
class MarketAwareLoss(nn.Module):
    """
    Combines multiple objectives for financial prediction.
    """
    def __init__(self, huber_delta=1.0, direction_weight=0.3, 
                 magnitude_weight=0.5, consistency_weight=0.2):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, pred, target):
        # 1. Magnitude loss (HuberLoss on absolute values)
        L_mag = self.huber(pred, target)
        
        # 2. Directional loss (did we get the sign right?)
        pred_dir = pred  # raw logit
        target_dir = (target > 0).float()
        L_dir = self.bce(pred_dir, target_dir)
        
        # 3. Temporal consistency loss (smooth predictions)
        #    Penalize large changes between consecutive predictions
        if pred.shape[0] > 1:
            L_consistency = torch.mean(torch.abs(pred[1:] - pred[:-1]))
        else:
            L_consistency = 0.0
        
        return (self.magnitude_weight * L_mag + 
                self.direction_weight * L_dir + 
                self.consistency_weight * L_consistency)
```

### 6.4 Advanced Optimizers

**Problem:** AdamW is a good default but may not be optimal for this non-stationary optimization landscape.

**Options to tune via Optuna:**

```
1. AdamW (current):         Stable, well-understood. Good default.
2. LAMB (Layer-wise Adaptive): Better for large batch training (batch>256).
3. Ranger (RAdam + Lookahead): Self-tuning LR + forward-looking param smoothing.
4. Adan:                    Nesterov-inspired momentum. Faster convergence on seq2seq.
5. Lion (Google Brain):     Sign-based optimizer. Memory-efficient (stores only momentum, not variance).
6. Sharpness-Aware Minimization (SAM): Seeks flat minima → better generalization.
   - SAM + AdamW: compute gradient, perturb weights, compute gradient again, step.
   - 2x compute per step but dramatically reduces overfitting.
   - Ideal for this use case (financial overfitting is the #1 risk).
```

**Recommended:** SAM + AdamW for final training (post-Optuna). Use AdamW for Optuna trials (speed), SAM for final model (generalization).

### 6.5 Learning Rate Schedule Improvements

```
Current options: cosine, cosine_warm, plateau
Missing options that should be added:

1. OneCycleLR (Smith's 1-Cycle Policy):
   - LR rises linearly from lr/25 to lr in first 30% of training
   - Then cosine decays to lr/10000
   - Often converges faster and to better minima
   - BEST for known epoch count (post-Optuna)

2. CosineAnnealingWarmRestarts with Decay:
   - Current: T_0 restarts without decay
   - Improved: multiply max_lr by 0.8 at each restart
   - Prevents the "forgetting" that can happen on restarts

3. Polynomial Decay:
   - LR = base_lr * (1 - step/total_steps)^power
   - Smoother than cosine, well-suited for long training
```

### 6.6 Gradient Accumulation for Effective Large Batch

**Problem:** RTX 3050 with 6GB limits batch_size to ~128. Larger effective batch sizes reduce gradient noise and can improve convergence.

**Solution:**

```python
# Accumulate gradients over K microbatches before stepping
accumulation_steps = 4  # effective batch = 128 * 4 = 512

for i, (x, y) in enumerate(loader):
    with torch.amp.autocast('cuda'):
        loss = criterion(model(x), y) / accumulation_steps  # scale loss
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### 6.7 Label Smoothing for Regression

**Problem:** Hard targets create sharp loss gradients. For noisy financial data, hard targets on exact range values may cause overconfident predictions.

**Solution — Regression Label Smoothing:**

```python
def smooth_targets(y, noise_scale=0.01):
    """Add small Gaussian noise to targets during training."""
    noise = torch.randn_like(y) * noise_scale * y.std()
    return y + noise
```

---

## 7. PHASE 5 — OVERFITTING & UNDERFITTING PREVENTION FRAMEWORK

### 7.1 Comprehensive Overfitting Detection

```
METRIC-BASED DETECTION:
┌─────────────────────────────────────────────────────────────┐
│ train_loss << val_loss → OVERFITTING                        │
│ Ratio: val_loss / train_loss > 1.3 → RED FLAG               │
│ Directional acc (train) >> Directional acc (val) → OVERFIT  │
│ CPCV combination variance > 5% → REGIME OVERFIT             │
│ Feature importance dominated by 1-2 features → FRAGILE      │
│                                                             │
│ train_loss ≈ val_loss AND both high → UNDERFITTING          │
│ Directional acc ≈ 50% on both splits → MODEL TOO SIMPLE    │
│ Loss plateaus early and never decreases → CAPACITY LIMITED  │
└─────────────────────────────────────────────────────────────┘

ACTION TRIGGERS:
  If overfit detected:
    1. Increase dropout (0.15 → 0.25 → 0.35)
    2. Increase weight_decay (1e-4 → 1e-3)
    3. Reduce hidden_size or num_layers
    4. Enable SAM optimizer
    5. Add data augmentation
    6. Reduce seq_len (less capacity for memorization)
    
  If underfit detected:
    1. Increase hidden_size (128 → 256 → 512)
    2. Add more attention layers (2 → 3 → 4)
    3. Increase epochs with lower early stopping patience
    4. Reduce dropout (0.15 → 0.05)
    5. Decrease weight_decay (1e-4 → 1e-6)
    6. Check feature quality (possibly insufficient signal)
```

### 7.2 Regularization Arsenal

All available regularization techniques ranked by priority:

```
TIER 1 — Already Implemented (keep/tune):
  ✅ Dropout (0.05-0.4)
  ✅ Weight Decay (L2, via AdamW: 1e-6 to 1e-2)
  ✅ Early Stopping (patience-based)
  ✅ EMA (Polyak averaging)
  ✅ Gradient Clipping (max_norm=1.0)

TIER 2 — Should Add (high priority):
  ☐ R-Drop (Regularized Dropout):
      Run same input through model twice with different dropout masks
      Minimize KL-divergence between two outputs
      Forces model to be robust to dropout patterns → reduces overfit
  
  ☐ Stochastic Depth (LayerDrop):
      Randomly skip entire attention layers during training (p=0.1-0.2)
      Equivalent to dropout for layers, not neurons
      Reduces co-adaptation between layers
  
  ☐ Spectral Normalization:
      Constrain weight matrix spectral norm to 1.0
      Prevents any single weight matrix from dominating
      σ(W) = max singular value → W_norm = W / σ(W)
  
  ☐ Mixup / CutMix for time series:
      Mixup: x_mix = λ*x_i + (1-λ)*x_j, y_mix = λ*y_i + (1-λ)*y_j
      Creates interpolated training samples
      Forces decision boundaries to be smooth
      λ ~ Beta(α, α), α = 0.2

TIER 3 — Should Add (medium priority):
  ☐ Data Augmentation (time-series specific):
      - Jittering: add small Gaussian noise to features
      - Scaling: multiply subsequences by random factor [0.8, 1.2]
      - Time warping: stretch/compress temporal segments
      - Window slicing: crop random sub-windows of sequences
      - Magnitude warping: smooth multiplicative perturbation
      
  ☐ Flooding Regularization:
      L_flood = |L - b| + b  where b is flood level
      Prevents loss from going below b, keeping model "searching" for flat minima
      Prevents overfitting by avoiding sharp loss landscapes
  
  ☐ Auxiliary Denoising Task:
      Corrupt 10% of input features with Gaussian noise
      Add reconstruction head: predict clean features from corrupted input
      Forces encoder to learn robust representations
```

### 7.3 Automatic Overfit/Underfit Diagnostic Report

After each training run, auto-generate a diagnostic:

```
╔══════════════════════════════════════════════════════════╗
║           OVERFIT/UNDERFIT DIAGNOSTIC REPORT            ║
╠══════════════════════════════════════════════════════════╣
║ Train Loss:    12.34                                    ║
║ Val Loss:      18.56                                    ║
║ Ratio:         1.50 [⚠ MILD OVERFIT — consider more    ║
║                       regularization]                    ║
║                                                          ║
║ Train Dir Acc: 62.3%                                    ║
║ Val Dir Acc:   51.2%                                    ║
║ Gap:           11.1% [⚠ OVERFIT — spurious patterns     ║
║                       memorized]                         ║
║                                                          ║
║ Best Epoch:    5 / 12                                   ║
║ Patience Used: 5/5 [✓ Early stopping triggered — good]  ║
║                                                          ║
║ Feature Importance Concentration:                        ║
║   Top-1 feature: 23% of total importance [⚠ fragile]    ║
║   Top-5 features: 71% of total importance [⚠ narrow]    ║
║                                                          ║
║ RECOMMENDATION: Increase dropout to 0.25,               ║
║   add weight_decay to 5e-4, enable R-Drop               ║
╚══════════════════════════════════════════════════════════╝
```

### 7.4 Noise Injection Robustness Test

After training, run a noise sensitivity analysis:

```
For noise_level in [0.01, 0.05, 0.10, 0.20]:
    x_noisy = x_val + noise_level * x_val.std() * N(0,1)
    metrics_noisy = evaluate(model, x_noisy, y_val)
    
    degradation = (metrics_clean - metrics_noisy) / metrics_clean * 100
    
    Report:
    | Noise Level | Val Loss Δ | Dir Acc Δ | Sharpe Δ |
    |-------------|-----------|----------|---------|
    | 1%          | +2.3%     | -0.5%    | -1.1%   |  ← Robust
    | 5%          | +8.7%     | -3.2%    | -5.4%   |  ← Acceptable
    | 10%         | +31.2%    | -11.4%   | -22.7%  |  ← Fragile ⚠
    | 20%         | +89.1%    | -28.3%   | -61.2%  |  ← Broken 🔴

If degradation at 5% noise > 15%: model is over-sensitive → needs more regularization
```

---

## 8. PHASE 6 — HYPERPARAMETER OPTIMIZATION EXPANSION

### 8.1 Expanded Search Space

```python
# CURRENT (narrow)                    # PROPOSED (institutional)
epochs:        6-14                    epochs:        3-5  (HARD MAX)
batch_size:    {32, 64, 128}          batch_size:    {32, 64, 128, 256, 512}
hidden_size:   {64, 128, 256}         hidden_size:   {64, 128, 256, 512}
dropout:       0.05-0.4               dropout:       0.02-0.5
lr:            1e-4 to 5e-3           lr:            5e-5 to 1e-2
weight_decay:  1e-6 to 1e-2           weight_decay:  1e-7 to 5e-2
seq_len:       {32, 64}               seq_len:       {16, 32, 64, 128}
horizon:       {10, 15, 30}           horizon:       {5, 10, 15, 30, 60}
num_heads:     {2, 4}                 num_heads:     {1, 2, 4, 8}
num_layers:    1-2                    num_layers:    1-6
use_attention: {T/F}                  use_attention: {T/F}
scheduler:     3 options              scheduler:     5 options (+onecycle, +poly)
warmup:        0-3                    warmup:        0-5
use_ema:       {T/F}                  use_ema:       {T/F}
                                      ema_decay:     0.99-0.9999
                                      
# NEW DIMENSIONS:
grad_clip_norm:      {0.5, 1.0, 2.0, 5.0}
huber_delta:         0.5-2.0
accumulation_steps:  {1, 2, 4}
use_sam:             {T/F}          # SAM optimizer
mixup_alpha:         0.0-0.4        # 0.0 = disabled
stochastic_depth:    0.0-0.2
scaler_type:         {standard, robust, quantile}

# XGB RESIDUAL SPACE (currently fixed!):
xgb_n_estimators:    200-2000
xgb_max_depth:       3-10
xgb_learning_rate:   0.005-0.1 (log)
xgb_subsample:       0.5-1.0
xgb_colsample:       0.5-1.0
xgb_reg_alpha:       1e-5 to 1.0 (log)
xgb_reg_lambda:      1e-3 to 10.0 (log)
xgb_min_child_weight: 1-10
xgb_gamma:           0.0-1.0
```

### 8.2 Multi-Objective Optimization

**Problem:** Optimizing only `best_val_loss` ignores directional accuracy, Sharpe ratio, and stability.

**Solution — Pareto-Front Optimization:**

```python
import optuna

study = optuna.create_study(
    directions=["minimize", "maximize", "maximize"],  # 3 objectives
    study_name="modelmk1_multi_objective",
    sampler=optuna.samplers.NSGAIISampler(),  # Multi-objective sampler
)

def objective(trial):
    # ... hyperparameter sampling ...
    metrics = run_training(args)
    
    val_loss = metrics["best_val_loss"]
    dir_acc = metrics["val_directional_accuracy"]
    sharpe = run_backtest(y_true, y_pred)["sharpe_ratio"]
    
    return val_loss, dir_acc, sharpe  # Pareto-optimize all three

# Select from Pareto front: balance between loss, accuracy, and risk-adjusted return
```

### 8.3 Optuna Sampler Comparison

Run a meta-study comparing samplers before main optimization:

```
1. TPESampler (current):    Good for moderate dimensions (~20 params)
2. CmaEsSampler:            Better for continuous params, poor for categorical
3. NSGAIISampler:           Multi-objective genetic algorithm
4. QMCSampler:              Quasi-Monte Carlo for initial exploration
5. BoTorchSampler:          Gaussian Process BO (best for low-dim, expensive trials)

Recommended protocol:
  - Phase A: 10 trials with QMCSampler (space-filling exploration)
  - Phase B: 50+ trials with TPESampler (exploitation)
  - Phase C: 20 trials with CmaEsSampler (continuous param refinement)
```

### 8.4 Joint LNN + XGB Optimization

**Problem:** Currently Optuna tunes only LNN hyperparameters; XGB uses fixed defaults. The two models must be jointly optimized since XGB quality depends on LNN residual distribution.

**Solution:**

```python
def joint_objective(trial):
    # LNN hyperparams
    lnn_params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        # ... etc
    }
    
    # XGB hyperparams (NEW)
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
    }
    
    # Train LNN → Get residuals → Train XGB on residuals → Evaluate hybrid
    lnn_metrics = run_lnn_training(lnn_params)
    hybrid_metrics = run_hybrid_training(xgb_params)
    
    return hybrid_metrics["hybrid_val_loss"]  # Optimize END-TO-END
```

---

## 9. PHASE 7 — EVALUATION & BACKTESTING RIGOR

### 9.1 Realistic Backtesting Engine

**Problem:** Current backtest uses `PnL = sign(pred) * y_true` — assumes perfect execution at mid price with zero cost. Real trading has slippage, commissions, and market impact.

**Solution — Institutional-Grade Backtest:**

```python
class RealisticBacktester:
    def __init__(self,
                 commission_bps: float = 2.0,        # 2 bps per side
                 slippage_bps: float = 1.0,           # 1 bp average slippage
                 market_impact_bps: float = 0.5,      # 0.5 bp for small size
                 max_position_size: float = 1.0,      # Normalized units
                 min_confidence: float = 0.6,          # Skip low-confidence predictions
                 max_consecutive_losses: int = 5,      # Stop trading after N losses
                 max_drawdown_pct: float = 0.10,       # 10% max drawdown circuit breaker
                 ramp_up_trades: int = 20):             # Paper-trade first N predictions
        ...
    
    def simulate(self, predictions, confidences, y_true, timestamps):
        equity = [initial_capital]
        trades = []
        
        for i in range(len(predictions)):
            # 1. Check circuit breakers
            if self._drawdown(equity) > self.max_drawdown_pct:
                break  # Stop trading
            if self._consecutive_losses(trades) >= self.max_consecutive_losses:
                self._cooldown(duration=24*60)  # 1-day cooldown
            
            # 2. Position sizing (Kelly criterion or confidence-weighted)
            if confidences[i] < self.min_confidence:
                continue  # Skip: too uncertain
            position_size = self._kelly_fraction(
                win_rate=self._rolling_win_rate(trades, window=50),
                avg_win=self._rolling_avg_win(trades, window=50),
                avg_loss=self._rolling_avg_loss(trades, window=50),
            )
            position_size = min(position_size, self.max_position_size)
            
            # 3. Execution simulation
            entry_price = self._apply_slippage(current_price, direction=sign(predictions[i]))
            commission = abs(position_size) * entry_price * self.commission_bps / 10000
            impact = abs(position_size) * entry_price * self.market_impact_bps / 10000
            
            # 4. PnL calculation
            raw_pnl = sign(predictions[i]) * y_true[i] * position_size
            net_pnl = raw_pnl - 2 * commission - impact  # commission on entry + exit
            
            equity.append(equity[-1] + net_pnl)
            trades.append(Trade(pnl=net_pnl, ...))
        
        return BacktestResult(equity, trades, metrics)
```

### 9.2 CPCV-Anchored Backtest Protocol

```
Instead of single train → backtest:

CPCV Sub-sampled Backtest (200 combinations from C(16,8)=12,870):
┌───────────────────────────────────────────────────────────────┐
│ Combo 1: Train [G1,G3,G5,G7,G9,G11,G13,G15] → Test [even G] │
│ Combo 2: Train [G2,G4,G6,G8,G10,G12,G14,G16] → Test [odd G] │
│ Combo 3: Train [G1,G2,G3,G4,G5,G6,G7,G8] → Test [G9-G16]   │
│ ...                                                           │
│ Combo 200: Train [random 8 groups] → Test [remaining 8]       │
└───────────────────────────────────────────────────────────────┘

Each combination:
  1. Purge + embargo at train/test boundaries
  2. Train model on purged train set
  3. Predict on all test group samples (strict OOS)
  4. Record metrics for this combination

Aggregation:
  - 200 independent OOS performance values → full distribution
  - Report: mean ± std, 95% CI, PBO estimate
  - If PBO > 0.30 → model is overfit, DO NOT deploy
  - This replaces single-point walk-forward estimates with
    statistically rigorous distributional estimates.
```

### 9.3 Extended Metrics Suite

```
Current Metrics (15):
  MSE, MAE, RMSE, Dir Acc, PnL, Max Drawdown,
  Win Rate, Profit Factor, Expectancy, Max Losses,
  Sharpe, Sortino, Calmar, #Trades, #Wins, #Losses

Add:
  RISK METRICS:
    - Value at Risk (VaR 95%, 99%): worst-case daily loss
    - Conditional VaR (CVaR / Expected Shortfall)
    - Tail Ratio: |mean positive tail| / |mean negative tail|
    - Ulcer Index: sqrt(mean(drawdown²)) — penalizes deep drawdowns more
  
  STABILITY METRICS:
    - Rolling Sharpe (60-day window): check for consistency
    - Sharpe Ratio of Sharpe Ratios: meta-consistency
    - Most recent 20% Sharpe vs. full-period Sharpe: detecting decay
    - Regime-conditioned metrics: Sharpe in trending vs. mean-reverting
  
  STATISTICAL SIGNIFICANCE:
    - Bootstrapped Sharpe confidence interval (1000 resamples)
    - Deflated Sharpe Ratio (López de Prado): adjusts for multiple testing
    - Probability of backtest overfitting (CSCV method)
    - p-value vs. random strategy (permutation test, 10000 shuffles)
  
  PREDICTIVE QUALITY:
    - Information Coefficient (IC): rank correlation of pred vs. actual
    - IC_IR: mean(IC) / std(IC) — IC consistency
    - Hit rate by confidence bucket (low/mid/high uncertainty)
    - Calibration plot: predicted magnitude vs. actual magnitude
```

### 9.4 Model Comparison Framework

```
For every model version, compute a standardized scorecard:

╔═══════════════════════════════════════════════════════════════╗
║                    MODEL SCORECARD v2.3.1                     ║
╠═══════════════════════════════════════════════════════════════╣
║ ACCURACY           │ Val Loss: 45.23  (↓12% vs v2.2)        ║
║                    │ Dir Acc:  57.3%  (↑4.1% vs v2.2)       ║
║                    │ IC:       0.08   (↑0.03)                ║
║────────────────────┼─────────────────────────────────────────║
║ ROBUSTNESS         │ Noise Sensitivity (5%): +6.2%           ║
║                    │ CPCV OOS Std: 3.1%                       ║
║                    │ Worst-Fold Dir Acc: 53.1%                ║
║────────────────────┼─────────────────────────────────────────║
║ RISK-ADJUSTED      │ Sharpe: 1.34 (net of costs)             ║
║                    │ Sortino: 1.87                            ║
║                    │ Max Drawdown: -8.2%                      ║
║                    │ VaR(99%): -2.1%                          ║
║────────────────────┼─────────────────────────────────────────║
║ STATISTICAL        │ Deflated Sharpe p-value: 0.023           ║
║                    │ Bootstrap 95% CI: [0.91, 1.77]           ║
║                    │ CSCV Overfit Prob: 12%                   ║
║────────────────────┼─────────────────────────────────────────║
║ EFFICIENCY         │ Train Time: 14m (CUDA)                   ║
║                    │ Inference: 0.3ms/sample                  ║
║                    │ Model Size: 2.4MB                        ║
║────────────────────┼─────────────────────────────────────────║
║ VERDICT: ✅ APPROVED for paper trading (conditional)          ║
║          Requires 30-day live validation before capital alloc  ║
╚═══════════════════════════════════════════════════════════════╝
```

### 9.5 Combinatorially Symmetric Cross-Validation (CSCV)

Detect probability of backtest overfitting (López de Prado method):

```
1. Split backtest into S equal-length sub-periods (S=16)
2. Form all combinations of S/2 sub-periods for "in-sample" (C(16,8) = 12870)
3. For each combination:
   a. Select best model config on in-sample sub-periods
   b. Evaluate the same config on out-of-sample sub-periods
   c. Rank the chosen config among all configs on OOS
4. Probability of Backtest Overfitting (PBO):
   PBO = fraction of combinations where chosen config ranks below median on OOS
5. If PBO > 0.5: the backtest is likely overfit → DO NOT DEPLOY
```

---

## 10. PHASE 8 — FAIL-SAFE & RESILIENCE ENGINEERING

### 10.1 Runtime Circuit Breakers

```python
class TradingCircuitBreaker:
    """
    Monitors model behavior in production and triggers safety stops.
    """
    
    RULES = {
        # PREDICTION ANOMALIES
        "prediction_explosion": {
            "condition": "|pred| > 5 * historical_std(preds)",
            "action": "reject_prediction",
            "alert": "WARNING"
        },
        "prediction_flatline": {
            "condition": "std(last_100_preds) < 0.01 * historical_std",
            "action": "queue_model_check",
            "alert": "WARNING"
        },
        "confidence_collapse": {
            "condition": "mean(last_50_confidences) < 0.3",
            "action": "pause_trading",
            "alert": "CRITICAL"
        },
        
        # PERFORMANCE DEGRADATION
        "rolling_accuracy_drop": {
            "condition": "rolling_100_dir_acc < 0.45",
            "action": "reduce_position_size_50pct",
            "alert": "WARNING"
        },
        "drawdown_breach": {
            "condition": "current_drawdown > max_allowed_drawdown",
            "action": "halt_all_trading",
            "alert": "CRITICAL"
        },
        "consecutive_loss_streak": {
            "condition": "consecutive_losses >= 7",
            "action": "pause_trading_24h",
            "alert": "WARNING"
        },
        
        # DATA ANOMALIES
        "input_distribution_shift": {
            "condition": "KL_div(current_features, training_features) > threshold",
            "action": "flag_for_retrain",
            "alert": "WARNING"
        },
        "missing_data_gap": {
            "condition": "time_since_last_tick > 5 * median_interval",
            "action": "skip_prediction",
            "alert": "INFO"
        },
        "feature_nan_inf": {
            "condition": "any(isnan(features) | isinf(features))",
            "action": "reject_prediction",
            "alert": "ERROR"
        },
        
        # SYSTEM HEALTH
        "gpu_memory_critical": {
            "condition": "gpu_free_memory < 500MB",
            "action": "switch_to_cpu_inference",
            "alert": "WARNING"
        },
        "inference_latency_spike": {
            "condition": "inference_time > 10 * median_inference_time",
            "action": "log_and_alert",
            "alert": "WARNING"
        }
    }
```

### 10.2 Graceful Degradation Pipeline

```
TIER 0 — Normal Operation:
  Full hybrid model (LNN + XGB residual)
  MC Dropout confidence estimation
  Position sizing by Kelly criterion
      ↓ (if LNN fails)
TIER 1 — LNN Fallback:
  XGBoost-only predictions (no hybrid)
  Reduced position sizing (50% max)
  Alert: "LNN unavailable, XGB-only mode"
      ↓ (if XGB also fails)
TIER 2 — Baseline Fallback:
  Simple momentum strategy: sign(SMA_5 - SMA_20)
  Minimal position sizing (25% max)
  Alert: "Model unavailable, heuristic mode"
      ↓ (if data feed fails)
TIER 3 — Flat:
  Close all positions
  No new trades
  Alert: "SYSTEM OFFLINE — manual intervention required"
```

### 10.3 Model Integrity Checks

```
PRE-INFERENCE VALIDATION (every prediction request):
  ✓ Model file exists and checksum matches training manifest
  ✓ Scaler file exists and feature count matches model input_size
  ✓ Input sequence length matches model's expected seq_len
  ✓ Input features are finite (no NaN/Inf)
  ✓ Input features within [mean - 10σ, mean + 10σ] of training distribution
  ✓ XGBoost model loaded and feature names match

POST-INFERENCE VALIDATION:
  ✓ Prediction is finite
  ✓ Prediction magnitude within 5σ of training target distribution
  ✓ Confidence score is in [0, 1]
  ✓ If prediction changed direction from previous by > 3σ, flag as suspicious
```

### 10.4 Data Drift Detection

```python
class DataDriftMonitor:
    """
    Compares live feature distributions to training reference.
    Uses Population Stability Index (PSI) and KL-divergence.
    """
    
    def __init__(self, training_features: np.ndarray, feature_names: list):
        self.reference_stats = {
            name: {
                "mean": col.mean(),
                "std": col.std(),
                "quantiles": np.percentile(col, [5, 25, 50, 75, 95]),
                "histogram": np.histogram(col, bins=20),
            }
            for name, col in zip(feature_names, training_features.T)
        }
    
    def check_drift(self, live_features: np.ndarray) -> dict:
        drift_report = {}
        for name, col in zip(self.feature_names, live_features.T):
            ref = self.reference_stats[name]
            
            # Population Stability Index
            psi = self._compute_psi(ref["histogram"], np.histogram(col, bins=20))
            
            # KS test (non-parametric distribution comparison)
            ks_stat, ks_pvalue = scipy.stats.ks_2samp(
                self._sample_reference(name, n=1000), col
            )
            
            drift_report[name] = {
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "drifted": psi > 0.2 or ks_pvalue < 0.01,
            }
        
        total_drifted = sum(1 for v in drift_report.values() if v["drifted"])
        if total_drifted > len(drift_report) * 0.3:
            return {"status": "CRITICAL_DRIFT", "details": drift_report}
        elif total_drifted > 0:
            return {"status": "MILD_DRIFT", "details": drift_report}
        return {"status": "STABLE", "details": drift_report}
```

### 10.5 Out-of-Distribution Detection

```
Method 1: Mahalanobis Distance
  - Compute Mahalanobis distance of new input to training distribution
  - If distance > χ²(p=0.99, df=n_features): flag as OOD
  - Fast, simple, assumes multivariate Gaussian

Method 2: Reconstruction Error
  - Train lightweight autoencoder on training feature representations
  - At inference: encode → decode → measure reconstruction MSE
  - If recon_error > 99th percentile of training recon_errors: flag as OOD
  
Method 3: Epistemic Uncertainty (already have MC Dropout)
  - If epistemic_uncertainty > 95th percentile of val_uncertainties: OOD
  - Already implemented! Just need threshold calibration

Action on OOD Detection:
  - Reduce position size proportionally to OOD score
  - Log for human review
  - If OOD persists for >60 minutes: trigger model retrain alert
```

---

## 11. PHASE 9 — INFERENCE & PRODUCTION HARDENING

### 11.1 ONNX Quantization Pipeline

```
Full Precision (FP32) → 2.4MB, 0.3ms/sample
    ↓
Dynamic Quantization (INT8) → ~0.6MB, 0.15ms/sample
  - Weight-only INT8 quantization via onnxruntime
  - No calibration data needed
  - Typical accuracy loss: < 0.5%
    ↓
Static Quantization (INT8) → ~0.6MB, 0.10ms/sample
  - Requires calibration dataset (100 samples from training)
  - Quantizes weights AND activations
  - Better accuracy than dynamic at same size
    ↓
FP16 Half Precision → ~1.2MB, 0.12ms/sample (on GPU)
  - Ideal for GPU inference
  - Negligible accuracy loss for this model scale
```

### 11.2 Model Serving Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVICE                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [Data Ingestion]                                          │
│    WebSocket/REST → Raw ticks → Feature pipeline           │
│    (same resample + indicators as training — CRITICAL)     │
│                                                            │
│  [Pre-Processing]                                          │
│    Scale with saved scaler → Build sequence                │
│    Validate: no NaN, no OOD, feature count matches         │
│                                                            │
│  [Model Ensemble]                                          │
│    ┌──────────────┐     ┌──────────────┐                   │
│    │  ONNX LNN    │────→│ XGBoost      │                   │
│    │ (GPU/CPU)    │     │ (CPU)        │                   │
│    └──────────────┘     └──────────────┘                   │
│    LNN_pred ────────────→ Residual + LNN = Hybrid_pred    │
│                                                            │
│  [Uncertainty Estimation]                                  │
│    MC Dropout (N=30) → Mean, Std → Confidence              │
│                                                            │
│  [Post-Processing]                                         │
│    Circuit breaker check                                   │
│    Drift monitor update                                    │
│    Position sizing recommendation                          │
│    Log prediction + metadata                               │
│                                                            │
│  [Output]                                                  │
│    {prediction, confidence, position_size, timestamp}       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 11.3 A/B Testing Infrastructure

```
Shadow Mode:
  - New model runs alongside current model
  - Predictions logged but NOT acted upon
  - After 30 days: compare Sharpe on LIVE data
  - If new_sharpe > old_sharpe * 1.1: promote to primary
  
Champion-Challenger:
  - Champion: current production model (90% of capital)
  - Challenger: new model candidate (10% of capital)
  - After 60 days: if challenger outperforms, swap roles
  - Prevents catastrophic model replacement
```

### 11.4 Model Versioning & Registry

```
outputs/
  model_registry/
    v1.0.0/
      lnn_best.pt
      xgb_residual.json
      lnn_scaler.joblib
      hybrid_manifest.json
      scorecard.json          ← metrics, train date, data range
      git_commit.txt          ← source code version
      config_snapshot.yaml    ← frozen training config
      optuna_best_params.json ← hyperparameters
    v1.1.0/
      ...
    v2.0.0/
      ...
    ACTIVE_VERSION.json → {"production": "v1.1.0", "shadow": "v2.0.0"}
```

---

## 12. PHASE 10 — INSTITUTIONAL INFRASTRUCTURE

### 12.1 Experiment Tracking Integration

```
Integration options (pick one):
  1. MLflow (OSS, self-hosted):
     - Log every training run: params, metrics, artifacts
     - Model registry with staging/production lifecycle
     - Compare runs visually
  
  2. Weights & Biases (cloud):
     - Real-time training dashboards
     - Hyperparameter importance analysis
     - Artifact versioning
  
  3. Lightweight (custom, minimal dependencies):
     - JSON-based experiment log in outputs/experiments/
     - Each run: {id, timestamp, params, metrics, artifacts_path, git_hash}
     - CLI: `python main.py compare v1.0.0 v2.0.0` → side-by-side metrics

Implementation in run_training:
  experiment = ExperimentTracker(name="modelmk1")
  experiment.log_params(args.__dict__)
  for epoch in range(epochs):
      ...
      experiment.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
  experiment.log_artifact(checkpoint_path)
  experiment.end_run()
```

### 12.2 Automated Retraining Pipeline

```
TRIGGER CONDITIONS:
  1. Scheduled: Weekly retrain on expanding data window
  2. Performance-based: Rolling Sharpe < threshold for 5 consecutive days
  3. Drift-based: PSI > 0.2 on >30% of features
  4. Data-volume: >50K new samples since last training

PIPELINE:
  ┌─────────────────┐
  │ Trigger fired    │
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Data validation  │ → Fail: Alert data team
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Feature pipeline │ → Cache if possible
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ CPCV validation  │ → If no improvement over current model: STOP
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Full training    │ → Optuna (10 trials) + best-param train + hybrid
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Scorecard check  │ → Must beat current model on ≥3/5 key metrics
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Deploy as shadow │ → 30-day live validation
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │ Promote if passes│ → Swap to production
  └─────────────────┘
```

### 12.3 Comprehensive Testing Pyramid

```
UNIT TESTS (current: partial → target: 90%+ coverage):
  ✅ indicators.py     ✅ loader.py       ✅ runtime.py
  ✅ lnn_model.py      ✅ backtest.py
  ☐ config.py          ☐ paths.py         ☐ xgb_model.py
  ☐ onnx_export.py     ☐ predict.py

INTEGRATION TESTS (currently: none → target: key flows):
  ☐ Full pipeline on synthetic data (100 samples)
  ☐ Optuna 2-trial run with pruning verification
  ☐ Hybrid training: verify residual reduction > 0%
  ☐ ONNX export + validation round-trip
  ☐ Predict → Backtest end-to-end

REGRESSION TESTS:
  ☐ Fixed-seed deterministic output check:
    Given seed=42, data="fixture_100_rows.parquet":
    assert model_output == expected_tensor (rtol=1e-5)
  ☐ Performance baseline: any code change must not degrade
    val_loss or dir_acc by more than 2% on reference dataset

STRESS TESTS:
  ☐ GPU OOM recovery: batch_size=4096 → catch and handle
  ☐ Corrupted checkpoint load → graceful error
  ☐ Missing data columns → clear error message
  ☐ Empty dataset → ValueError with instructions
  ☐ NaN-only features → detected and logged

PROPERTY-BASED TESTS (hypothesis/pytest):
  ☐ For any valid OHLCV dataframe: add_all_indicators returns no NaN
  ☐ For any sequence: model forward pass returns (B, output_size)
  ☐ For any scaler fit + transform: inverse_transform recovers original (rtol=1e-5)
  ☐ For any non-empty predictions: all backtest metrics are finite
```

### 12.4 Documentation Standards

```
Required documentation for institutional sign-off:

1. MODEL CARD (per Google's Model Cards format):
   - Model name and version
   - Intended use and scope
   - Training data description (date range, source, size)
   - Performance metrics (with confidence intervals)
   - Ethical considerations (market manipulation risk)
   - Limitations and failure modes
   - Update/retrain frequency

2. DATA SHEET:
   - Data source and collection method
   - Feature definitions and units
   - Known biases (survivorship bias, look-ahead bias check)
   - Data quality metrics (missing rate, invalid rate)

3. RISK ASSESSMENT:
   - Maximum loss scenarios
   - Model failure modes and mitigations
   - Regulatory compliance considerations
   - Audit trail for model decisions

4. OPERATIONAL RUNBOOK:
   - How to retrain the model
   - How to roll back to a previous version
   - Incident response for model failures
   - Monitoring dashboard access and alert escalation
```

---

## 13. PHASE 11 — ADVANCED TECHNIQUES & RESEARCH FRONTIER

### 13.1 Knowledge Distillation for Low-Latency Inference

```
Teacher: Full MarketLNN (attention + GRU + SE) — slow, accurate
Student: Lightweight MLP or small GRU — fast, compact

Training:
  1. Train teacher on full dataset → teacher_preds
  2. Student loss = α * MSE(student, y_true) + (1-α) * MSE(student, teacher_preds)
  3. Student learns from both ground truth AND teacher's dark knowledge
  4. Typical: student achieves 95% of teacher accuracy at 10x inference speed

Use case: Low-latency live trading where <1ms inference is critical
```

### 13.2 Continual Learning (Avoid Catastrophic Forgetting)

```
Problem: When retraining on new data, model forgets patterns from old data.

Solution — Elastic Weight Consolidation (EWC):
  1. After training on Task_1 (2022-2024 data):
     - Compute Fisher Information Matrix for each parameter
     - F_i = E[(d log p(y|x,θ) / dθ_i)²] — how important is each param?
  2. When training on Task_2 (2024-2026 data):
     - New loss = L_task2 + λ * Σᵢ F_i * (θ_i - θ*_task1)²
     - Penalizes changing parameters that were important for Task_1
  3. Model learns new patterns while preserving critical old patterns

Alternative — Replay Buffer:
  - Maintain 5% random sample of old training data
  - Mix old samples (10%) with new samples (90%) during retraining
  - Simple, effective, no Fisher computation needed
```

### 13.3 Adversarial Training for Robustness

```
Principle: Train model to be robust against worst-case input perturbations.

Fast Gradient Sign Method (FGSM) for time series:
  1. Compute gradient of loss w.r.t. input: g = ∂L/∂x
  2. Create adversarial example: x_adv = x + ε * sign(g)
  3. Train on both clean and adversarial: L = L(x) + L(x_adv)

Adversarial budget ε:
  - Small: 0.01 * x.std() (barely perceptible noise)
  - Medium: 0.05 * x.std() (moderate perturbation)
  - Use medium for training, test on large perturbations

Benefits:
  - Model becomes robust to noisy real-world data
  - Reduces sensitivity to exact feature values
  - Acts as strong regularization (prevents overfitting to sharp features)
```

### 13.4 Self-Supervised Pre-Training

```
Pre-train the LNN encoder on UNLABELED data before supervised training.

Task 1: Masked Feature Prediction
  - Randomly zero out 15% of input features
  - Train to reconstruct masked features
  - Forces encoder to learn feature relationships

Task 2: Contrastive Learning (SimCLR for time series)
  - Augment each sequence twice (jitter, scale, warp)
  - Train: embeddings of same sequence should be similar
  - Embeddings of different sequences should be dissimilar
  - InfoNCE loss

Task 3: Next-Step Prediction
  - Given x[1:T-1], predict x[T]
  - Self-supervised (no labels needed)
  - Forces model to learn temporal dynamics

Protocol:
  1. Pre-train encoder on ALL available data (labeled + unlabeled + other assets)
  2. Freeze encoder weights
  3. Fine-tune prediction head on labeled data
  4. Optionally: unfreeze encoder with 10x lower learning rate
```

### 13.5 Conformal Prediction for Guaranteed Coverage

```
Problem: MC Dropout uncertainty has no theoretical guarantees.
         Confidence = 1/(1+std) is heuristic, not calibrated.

Solution — Split Conformal Prediction:
  1. Hold out calibration set (10% of validation data)
  2. Compute residuals: r_i = |y_i - ŷ_i| for each calibration sample
  3. Set quantile level α (e.g., 0.05 for 95% coverage)
  4. Compute threshold: q = quantile(r, (1-α)(1+1/n))
  5. Prediction interval: [ŷ - q, ŷ + q]
  
  GUARANTEE: P(y ∈ [ŷ-q, ŷ+q]) ≥ 1-α (regardless of model!)
  
Adaptive Conformal:
  - Use locally-weighted quantiles (wider intervals for uncertain regions)
  - Or: use model's uncertainty to modulate interval width
  - Interval = [ŷ - q * σ_epistemic, ŷ + q * σ_epistemic]
```

### 13.6 Ensemble of Diverse Models

```
Instead of single LNN + XGB:

MODEL POOL:
  1. MarketLNN (attention + GRU) — current architecture
  2. MarketLNN (attention-only, no GRU) — pure transformer
  3. MarketLNN (GRU-only, no attention) — pure recurrent
  4. Temporal Convolutional Network (TCN) — dilated causal conv
  5. XGBoost (direct, not residual) — fully tabular
  6. LightGBM (gradient-based) — faster training
  7. CatBoost (ordered boosting) — handles categoricals natively

ENSEMBLE STRATEGY:
  Method 1 — Simple Average (baseline):
    pred = mean([model_i(x) for i in models])
  
  Method 2 — Weighted Average (learned):
    weights = softmax(Linear(val_metrics))  # weight by inverse val_loss
    pred = sum(w_i * model_i(x))
  
  Method 3 — Stacking (meta-learner):
    base_preds = [model_i(x) for i in models]  # First-level predictions
    meta_model = Ridge().fit(base_preds_train, y_train)  # Second level
    pred = meta_model.predict(base_preds)
  
  Method 4 — Uncertainty-Weighted:
    pred = sum(confidence_i * model_i(x)) / sum(confidence_i)
    (Models that are more confident get more weight)

DIVERSITY ENFORCEMENT:
  - Train each model variant on slightly different feature subsets
  - Use different random seeds for each
  - Use different data splits (bootstrap sampling)
  - Penalize correlated predictions during stacking training
```

### 13.7 Causal Inference Integration

```
Move beyond correlation to causation:

1. Granger Causality Testing:
   - For each feature: does it Granger-cause the target?
   - Keep only features that pass at p < 0.05
   - Removes spurious correlations (e.g., price + SMA are 98% correlated
     but SMA doesn't cause price — it follows it)

2. Instrumental Variable Approach:
   - Use volume as instrument for price impact studies
   - Disentangle: does high volume cause future range expansion?

3. Causal Discovery (PC Algorithm or NOTEARS):
   - Build Directed Acyclic Graph (DAG) of feature relationships
   - Identify true causal parents of target
   - Use only causal features for prediction

4. Counterfactual Analysis:
   - "What would the prediction be if RSI had been 30 instead of 70?"
   - Requires structural causal model from step 3
   - Enables what-if scenario analysis
```

---

## 14. RETAIL-ALGORITHMIC MODEL LIMITATION MATRIX

Every limitation commonly faced by retail algo systems, mapped to specific solutions in this plan:

| # | Retail Limitation | Status in MK1 | Solution Phase | Fix |
|---|------------------|---------------|----------------|-----|
| 1 | No walk-forward validation → overfitting to past | **PRESENT** | Phase 4 §6.1 | Combinatorial Purged Cross-Validation (CPCV) with PBO estimation |
| 2 | Unrealistic backtesting (no slippage/fees) | **PRESENT** | Phase 7 §9.1 | RealisticBacktester with commissions, slippage, impact |
| 3 | Optimizing in-sample then reporting in-sample metrics | **PRESENT** | Phase 7 §9.2 | Walk-forward OOS-only evaluation |
| 4 | No statistical significance testing | **PRESENT** | Phase 7 §9.3 | Deflated Sharpe, bootstrap CI, permutation tests |
| 5 | Overfitting to hyperparameters (multiple testing) | **PRESENT** | Phase 7 §9.5 | CSCV backtest overfit probability |
| 6 | Single model for all market regimes | **PRESENT** | Phase 3 §5.2 | Mamba SSM with input-dependent selective state transitions for implicit regime adaptation |
| 7 | No concept drift / data drift detection | **PRESENT** | Phase 8 §10.4 | PSI + KL-divergence monitoring |
| 8 | No position sizing / risk management | **PRESENT** | Phase 7 §9.1 | Kelly criterion + confidence-weighted sizing |
| 9 | No circuit breakers / drawdown limits | **PRESENT** | Phase 8 §10.1 | Full circuit breaker framework |
| 10 | Feature engineering by gut feel | **PARTIAL** | Phase 2 §4.4 | Mutual information + permutation importance pipeline |
| 11 | Non-stationary features (raw price) | **PRESENT** | Phase 2 §4.5 | ADF testing + automatic differencing |
| 12 | No model versioning / rollback | **PRESENT** | Phase 9 §11.4 | Version registry with active version tracking |
| 13 | Manual retraining (forget or delay) | **PRESENT** | Phase 10 §12.2 | Automated retrain triggers |
| 14 | Black-box predictions | **PRESENT** | Phase 2 §4.4 + Phase 3 §5.3 | SHAP, attention viz, variable selection networks |
| 15 | Survivorship bias in data | **UNKNOWN** | Phase 1 §3.4 | Data quality gates + documentation |
| 16 | Look-ahead bias in feature construction | **ABSENT** | Phase 1 §3.4 | Explicit leakage checks per feature |
| 17 | Overfit to specific asset / timeframe | **LIKELY** | Phase 4 §6.1 | Multi-fold, multi-regime validation |
| 18 | No graceful degradation | **PRESENT** | Phase 8 §10.2 | 4-tier fallback system |
| 19 | Ignoring transaction costs in optimization | **PRESENT** | Phase 6 §8.2 | Multi-objective with net-of-cost Sharpe |
| 20 | Single point predictions (no uncertainty) | **PARTIAL** (MC only) | Phase 3 §5.5 | Aleatoric + epistemic + conformal prediction |
| 21 | No out-of-distribution detection | **PRESENT** | Phase 8 §10.5 | Mahalanobis + reconstruction + epistemic methods |
| 22 | No ensemble diversity | **PRESENT** | Phase 11 §13.6 | Multi-architecture ensemble with stacking |
| 23 | Training on all data equally | **PRESENT** | Phase 4 §6.2 | Curriculum learning + sample weighting |
| 24 | Fixed indicator parameters | **PRESENT** | Phase 2 §4.1 | CNN stem for learned features |
| 25 | No pre-training (data-hungry from scratch) | **PRESENT** | Phase 11 §13.4 | Self-supervised pre-training tasks |
| 26 | Catastrophic forgetting on retrain | **PRESENT** | Phase 11 §13.2 | EWC + replay buffer |
| 27 | No adversarial robustness | **PRESENT** | Phase 11 §13.3 | FGSM adversarial training |
| 28 | Model too large for low-latency | **MINOR** | Phase 9 §11.1 + §13.1 | ONNX quantization + knowledge distillation |
| 29 | XGB hyperparams never tuned | **PRESENT** | Phase 6 §8.4 | Joint LNN + XGB Optuna optimization |
| 30 | No A/B testing framework | **PRESENT** | Phase 9 §11.3 | Champion-challenger protocol |

---

## 15. IMPLEMENTATION PRIORITY MATRIX

Phases ordered by **impact × feasibility**, accounting for dependencies:

### IMMEDIATE (Week 1-2): Foundation Fixes

| Priority | Item | Impact | Effort | Dependency |
|----------|------|--------|--------|------------|
| **P0** | CPCV Validation (§6.1) | 🔴 Critical | Medium | None |
| **P0** | Multi-Task Target Decoupling (§3.2) | 🔴 Critical | Medium | None |
| **P0** | Feature Caching Layer (§3.5) | 🟡 High | Low | None |
| **P0** | Adaptive Scaler (§3.1) | 🟡 High | Low | None |
| **P0** | Data Quality Gates (§3.4) | 🟡 High | Low | None |

### SHORT-TERM (Week 3-4): Accuracy & Robustness

| Priority | Item | Impact | Effort | Dependency |
|----------|------|--------|--------|------------|
| **P1** | Composite Loss Function (§6.3) | 🔴 Critical | Medium | §3.2 |
| **P1** | Joint LNN + XGB Optuna (§8.4) | 🔴 Critical | Medium | None |
| **P1** | Expanded Search Space (§8.1) | 🟡 High | Low | None |
| **P1** | Feature Selection Pipeline (§4.4) | 🟡 High | Medium | None |
| **P1** | Feature Stationarity (§4.5) | 🟡 High | Low | None |
| **P1** | Overfit/Underfit Diagnostic (§7.3) | 🟡 High | Low | None |

### MID-TERM (Week 5-8): Architecture & Training

| Priority | Item | Impact | Effort | Dependency |
|----------|------|--------|--------|------------|
| **P2** | RoPE Positional Encoding (§5.1.2) | 🟡 High | Low | None |
| **P2** | SwiGLU FeedForward (§5.1.3) | 🟡 High | Low | None |
| **P2** | Aleatoric Uncertainty Head (§5.5) | 🟡 High | Medium | §3.2 |
| **P2** | SAM Optimizer (§6.4) | 🟡 High | Low | None |
| **P2** | OneCycleLR Scheduler (§6.5) | 🟢 Medium | Low | None |
| **P2** | Gradient Accumulation (§6.6) | 🟢 Medium | Low | None |
| **P2** | Noise Robustness Test (§7.4) | 🟢 Medium | Low | None |
| **P2** | R-Drop Regularization (§7.2) | 🟢 Medium | Low | None |

### LONG-TERM (Week 9-16): Production & Advanced

| Priority | Item | Impact | Effort | Dependency |
|----------|------|--------|--------|------------|
| **P3** | Realistic Backtester (§9.1) | 🔴 Critical | High | None |
| **P3** | Circuit Breakers (§10.1) | 🔴 Critical | Medium | None |
| **P3** | Data Drift Monitor (§10.4) | 🟡 High | Medium | None |
| **P3** | Model Versioning (§11.4) | 🟡 High | Medium | None |
| **P3** | Statistical Significance Tests (§9.3) | 🟡 High | Medium | §6.1 |
| **P3** | Experiment Tracking (§12.1) | 🟢 Medium | Medium | None |
| **P3** | Integration Tests (§12.3) | 🟢 Medium | Medium | None |

### RESEARCH (Week 16+): Frontier Techniques

| Priority | Item | Impact | Effort | Dependency |
|----------|------|--------|--------|------------|
| **P4** | Mamba SSM Integration (§5.2) | 🟡 High | Medium | §5.1 |
| **P4** | InceptionTime Feature Extraction (§4.1) | 🟡 High | Medium | §4.4 |
| **P4** | Self-Supervised Pre-Training (§13.4) | 🟡 High | High | §5.1 |
| **P4** | Conformal Prediction (§13.5) | 🟡 High | Medium | §5.5 |
| **P4** | Multi-Model Ensemble (§13.6) | 🟡 High | High | P3 complete |
| **P4** | Continual Learning (§13.2) | 🟢 Medium | High | §12.2 |
| **P4** | Knowledge Distillation (§13.1) | 🟢 Medium | Medium | Full model trained |
| **P4** | Adversarial Training (§13.3) | 🟢 Medium | Medium | None |
| **P4** | Causal Inference (§13.7) | 🟢 Medium | High | §4.4 |

---

## 16. APPENDIX: TECHNIQUE REFERENCE

### Loss Functions Reference

| Loss | Formula | When to Use |
|------|---------|-------------|
| MSE | $(y - \hat{y})^2$ | Standard regression, no outliers |
| MAE | $\|y - \hat{y}\|$ | Robust to outliers, median estimation |
| Huber | MSE if $\|e\| < \delta$, MAE otherwise | Current default — good baseline |
| LogCosh | $\log(\cosh(y - \hat{y}))$ | Smoother than Huber, differentiable everywhere |
| Quantile | $\max(q \cdot e, (q-1) \cdot e)$ | Prediction intervals, asymmetric loss |
| Gaussian NLL | $\frac{1}{2}\log\sigma^2 + \frac{(y-\mu)^2}{2\sigma^2}$ | Aleatoric uncertainty estimation |

### Optimizer Reference

| Optimizer | Key Property | Best For |
|-----------|-------------|----------|
| AdamW | Decoupled weight decay | Default choice |
| SAM | Flat-minima seeking | Generalization (anti-overfit) |
| Lion | Sign-based momentum | Memory-efficient, faster |
| LAMB | Layer-wise adaptive | Large batch training |
| Adan | Nesterov momentum | Sequential models |

### Regularization Reference

| Technique | Type | Anti-Overfit Strength |
|-----------|------|----------------------|
| Dropout | Stochastic | ★★★☆☆ |
| Weight Decay | Penalty | ★★☆☆☆ |
| EMA | Averaging | ★★☆☆☆ |
| R-Drop | Consistency | ★★★★☆ |
| SAM | Optimization | ★★★★☆ |
| Mixup | Augmentation | ★★★☆☆ |
| Stochastic Depth | Stochastic | ★★★☆☆ |
| Spectral Norm | Constraint | ★★★☆☆ |
| Flooding | Landscape | ★★★★☆ |
| Adversarial | Augmentation | ★★★★★ |

---

## 17. INFERENCE SPEED ANALYSIS — 1-MINUTE AND 5-MINUTE BARS

### 17.1 FLOP-Level Forward Pass Breakdown (MarketLNN)

Architecture: Input projection → LayerNorm → Positional Encoding → Squeeze-Excite → N×(TemporalAttention + FeedForward) → GRU backbone → Prediction Head

Configuration: `input_size=44, hidden_size=128, num_layers=2, num_heads=4, seq_len=64, gru_layers=2`

```
COMPONENT-BY-COMPONENT FLOP COUNT:
─────────────────────────────────────────────────────────────────────────
Component                         FLOPs           Notes
─────────────────────────────────────────────────────────────────────────
1. Linear Projection (44→128)     44×128×64        = 360,448
   Per timestep: 44×128 = 5,632
   × 64 timesteps

2. LayerNorm                      128×64×4         = 32,768
   (mean, var, normalize, scale)

3. Positional Encoding            128×64           = 8,192
   (addition only)

4. Squeeze-Excite                                  = 540,672
   Global avg pool: 128×64                = 8,192
   FC down: 128×32 = 4,096                = 4,096
   ReLU: 32                               = 32
   FC up: 32×128 = 4,096                  = 4,096
   Sigmoid: 128                            = 128
   Scale: 128×64                           = 8,192
   Subtotal per application                = 24,736
   (applied once post-projection)

5. TemporalAttention (×2 layers)                   = 2,228,224
   Per layer:
     Q projection: 128×128×64             = 1,048,576 → WAIT
     ─── Corrected ───
     Q,K,V: 3 × (B,T,C)@(C,C) = 3×128²  = 49,152 per timestep
     × 64 timesteps = 3,145,728 → NO, matrix multiply:
     
   Q = X @ W_Q: (64,128) @ (128,128)     = 64×128×128    = 1,048,576
   K = X @ W_K: same                      = 1,048,576
   V = X @ W_V: same                      = 1,048,576
   Attention scores: Q @ K^T / √d         = 64×64×128     = 524,288
   Softmax: 64×64×4 (per head)            = 16,384
   Attn @ V: (64,64) @ (64,128)           = 524,288
   Output proj: 128×128×64                = 1,048,576
   Residual + LayerNorm                    = 32,768
   FeedForward: 128→512→128 (×64)         = 2×128×512×64  = 8,388,608
   FF Residual + LayerNorm                 = 32,768
   ─────────────────────────
   Total per layer                         ≈ 12,665,632
   × 2 layers                             = 25,331,264

6. GRU Backbone (2 layers, hidden=128)             = 25,165,824
   Per timestep per layer:
     3 gates × (input + hidden) × hidden
     = 3 × (128+128) × 128                = 98,304
   × 64 timesteps × 2 layers              = 12,582,912 × 2

7. Prediction Head (128→64→32→1)                   = 10,272
   FC1: 128×64                             = 8,192
   FC2: 64×32                              = 2,048
   FC3: 32×1                               = 32

8. Total Forward Pass FLOPs:                        ≈ 53.3M FLOPs
─────────────────────────────────────────────────────────────────────────
   Projection:      360K    (0.7%)
   Attention:        25.3M  (47.5%)  ← dominates (O(T²) attention)
   GRU:             25.2M  (47.2%)  ← second largest
   Head:            10K     (0.02%)
   SE + Norm:       582K    (1.1%)
   
   Note: With Mamba SSM replacing Attention+GRU:
   Mamba FLOPs ≈ 8-12M (O(n) linear) → TOTAL ≈ 12-16M FLOPs
   → 3.5-4.5× faster than current architecture
```

### 17.2 Inference Latency by Hardware Configuration

```
MEASURED / ESTIMATED LATENCY — SINGLE SAMPLE INFERENCE:
═══════════════════════════════════════════════════════════════════
Configuration              Latency (ms)     Throughput (samples/s)
═══════════════════════════════════════════════════════════════════
RTX 3050 — FP32 (current)   0.3 - 0.6       1,700 - 3,300
RTX 3050 — AMP FP16         0.15 - 0.35     2,800 - 6,600
RTX 3050 — TensorRT FP16    0.08 - 0.20     5,000 - 12,500
CPU (Ryzen 7) — FP32        1.5 - 3.0       330 - 660
ONNX Runtime CPU — FP32     0.8 - 1.5       660 - 1,250
ONNX Runtime CPU — INT8     0.3 - 0.7       1,400 - 3,300
ONNX Runtime GPU — FP16     0.10 - 0.25     4,000 - 10,000
───────────────────────────────────────────────────────────────────
With Mamba SSM (projected):
RTX 3050 — FP32             0.10 - 0.25     4,000 - 10,000
RTX 3050 — AMP FP16         0.05 - 0.15     6,600 - 20,000
═══════════════════════════════════════════════════════════════════

XGBoost residual inference:  0.01 - 0.05 ms (CPU, single sample)
MC Dropout (30 passes):      30 × single = 9.0 - 18.0 ms (GPU FP32)
                             30 × single = 4.5 - 10.5 ms (GPU FP16)
```

### 17.3 Full Pipeline Budget — 1-Minute Bars (60,000 ms available)

```
FULL PREDICTION PIPELINE TIMING:
═══════════════════════════════════════════════════════════════════════
Stage                              Time (ms)    % of 60s Budget
═══════════════════════════════════════════════════════════════════════
1. Data ingestion (WebSocket/REST)    1 - 5         0.008%
   Parse incoming bar, append to buffer

2. OHLCV resampling (if raw ticks)    2 - 10        0.017%
   Aggregate ticks → 1-min OHLCV bar

3. Feature computation (44 indicators)               
   a. Rolling indicators (SMA, EMA,    5 - 20        0.033%
      RSI, MACD, ATR, Bollinger)
   b. Volume indicators (VWAP, OBV,    2 - 8         0.013%
      CMF, volume z-score)
   c. Directional (ADX, momentum,      2 - 5         0.008%
      returns, Parkinson vol)
   d. Feature normalization             1 - 3         0.005%
   Subtotal features:                  10 - 36        0.060%

4. Sequence construction               0.1 - 0.5     0.001%
   Slice last 64 bars from buffer
   Apply pre-fitted scaler

5. LNN forward pass (GPU FP16)         0.15 - 0.35   0.001%
   Single deterministic prediction

6. XGBoost residual prediction          0.01 - 0.05   0.000%
   Hybrid = LNN_pred + XGB_residual

7. MC Dropout uncertainty (30 passes)   4.5 - 10.5    0.018%
   Mean ± std → confidence score

8. Post-processing                      0.5 - 1.0     0.002%
   Circuit breaker check, drift update,
   position sizing, logging
═══════════════════════════════════════════════════════════════════════
TOTAL PIPELINE (with MC Dropout):      18 - 63 ms     0.105%
TOTAL PIPELINE (single pass):          14 - 53 ms     0.088%
═══════════════════════════════════════════════════════════════════════

REMAINING TIME BUDGET:                 59,937 ms      99.9%
→ Available for: order management, risk checks, execution, monitoring
→ VERDICT: ✅ EASILY within 1-minute budget — pipeline uses <0.11%
→ Could run 950+ sequential predictions per minute if needed
```

### 17.4 Full Pipeline Budget — 5-Minute Bars (300,000 ms available)

```
FULL PREDICTION PIPELINE TIMING:
═══════════════════════════════════════════════════════════════════════
Stage                              Time (ms)    % of 300s Budget
═══════════════════════════════════════════════════════════════════════
1. Data ingestion                     1 - 5         0.002%
2. OHLCV resampling                   2 - 15        0.005%
   (aggregate 5 min of raw data)
3. Feature computation                10 - 36        0.012%
4. Sequence construction              0.1 - 0.5     0.000%
5. LNN forward pass (GPU FP16)       0.15 - 0.35   0.000%
6. XGBoost residual prediction        0.01 - 0.05   0.000%
7. MC Dropout (30 passes)             4.5 - 10.5    0.004%
8. Post-processing                    0.5 - 1.0     0.000%
═══════════════════════════════════════════════════════════════════════
TOTAL PIPELINE:                       18 - 68 ms     0.023%
═══════════════════════════════════════════════════════════════════════

REMAINING TIME BUDGET:                299,932 ms     99.98%
→ VERDICT: ✅ TRIVIALLY within 5-minute budget — uses <0.023%
→ Could run 4,400+ sequential predictions per 5-min window

ADDITIONAL CAPACITY PER 5-MIN WINDOW:
  - Full CPCV retrain (1 combination): ~40-120 seconds (feasible!)
  - Multi-asset scan (10 assets): ~180-680 ms
  - Ensemble of 7 models: ~126-441 ms
  - 100-pass MC Dropout for high-confidence: ~15-35 ms
```

### 17.5 Inference Speed Summary

```
KEY CONCLUSIONS:
═══════════════════════════════════════════════════════════════════════
1. CURRENT ARCHITECTURE (MarketLNN + XGBoost):
   - Single prediction: 0.3 ms (GPU) / 1.5 ms (CPU)
   - Full pipeline with MC Dropout: 18-63 ms
   - Uses <0.11% of 1-min budget, <0.023% of 5-min budget
   - INFERENCE IS NOT A BOTTLENECK — feature computation dominates

2. WITH MAMBA SSM (projected):
   - Single prediction: 0.10 ms (GPU) / 0.5 ms (CPU)
   - Full pipeline with MC Dropout: 12-45 ms
   - 1.5-2× faster model inference (but features still dominate)

3. OPTIMIZATION PRIORITIES (if latency ever matters):
   Rank 1: Cache features (avoid recomputation) → saves 10-36 ms
   Rank 2: ONNX export + TensorRT → saves ~0.2 ms per pass
   Rank 3: Reduce MC Dropout passes (30→10) → saves 6-14 ms
   Rank 4: Mamba SSM replacement → saves ~0.2 ms per pass
   Rank 5: INT8 quantization → saves ~0.1 ms per pass

4. REAL-TIME CAPABILITY:
   1-min bars: ✅ 950+ predictions per minute capacity
   5-min bars: ✅ 4,400+ predictions per window capacity
   Tick-level (100ms): ✅ Feasible with cached features + single pass
═══════════════════════════════════════════════════════════════════════
```

---

## 18. AGGRESSIVE OPTUNA EARLY PRUNING STRATEGY

### 18.1 Philosophy — Kill Fast, Fail Cheap

Standard Optuna pruning wastes compute on bad trials. With a 5-epoch hard maximum per trial, every epoch is precious. The pruning strategy must be **ruthless**: identify hopeless trials within 1-3 epochs and free resources for promising configurations.

```
GUIDING PRINCIPLE:
  - Epoch 1: Sanity check — is this trial obviously broken?
  - Epoch 2: Active comparison — is it competitive?
  - Epoch 3: HARD KILL ZONE — converging or dead
  - Epoch 4-5: ONLY survivors with clear convergence trajectory
  - ABSOLUTE MAX: 5 epochs. No exceptions. No extensions.
```

### 18.2 Epoch-by-Epoch Pruning Protocol

```
EPOCH 1 — SANITY GATE:
  ┌─────────────────────────────────────────────────────────┐
  │ IF val_loss > 3.0 × best_known_loss THEN:              │
  │   → IMMEDIATE PRUNE (configuration is catastrophic)     │
  │   → Example: best_known = 45.0, trial gets 150+ → dead │
  │                                                         │
  │ IF val_loss == NaN or Inf THEN:                         │
  │   → IMMEDIATE PRUNE (numerical instability)             │
  │                                                         │
  │ ELSE: SURVIVE → proceed to epoch 2                      │
  └─────────────────────────────────────────────────────────┘

EPOCH 2 — COMPETITIVE GATE:
  ┌─────────────────────────────────────────────────────────┐
  │ Collect: median val_loss of ALL completed trials         │
  │                                                         │
  │ IF val_loss > 1.5 × median_completed THEN:             │
  │   → PRUNE (clearly uncompetitive)                       │
  │                                                         │
  │ IF val_loss > epoch_1_loss (INCREASING loss):           │
  │   → PRUNE (diverging — wrong learning rate or config)   │
  │                                                         │
  │ IF Δloss < 0.5% improvement from epoch 1:              │
  │   → YELLOW FLAG (borderline — one more chance)          │
  │                                                         │
  │ ELSE: SURVIVE → proceed to epoch 3                      │
  └─────────────────────────────────────────────────────────┘

EPOCH 3 — HARD KILL ZONE (most trials die here):
  ┌─────────────────────────────────────────────────────────┐
  │ IF no improvement over BEST of epoch 1 & 2:            │
  │   → PRUNE (stagnant — will not converge in 2 epochs)   │
  │                                                         │
  │ IF val_loss still > 1.2 × median_completed:            │
  │   → PRUNE (still uncompetitive after 3 epochs)         │
  │                                                         │
  │ IF val_loss > 1.0 × best_trial_at_epoch_3:             │
  │   → PRUNE (inferior to existing best at same stage)     │
  │                                                         │
  │ IF cumulative improvement < 5% from epoch 1:           │
  │   → PRUNE (insufficient convergence rate)               │
  │                                                         │
  │ ELSE: SURVIVE → proceed to epoch 4 (rare — ~20% of     │
  │       trials should reach this point)                    │
  └─────────────────────────────────────────────────────────┘

EPOCH 4 — CONVERGENCE CONFIRMATION:
  ┌─────────────────────────────────────────────────────────┐
  │ Only survivors from epoch 3 reach here                  │
  │                                                         │
  │ IF val_loss plateaued (Δ < 0.1% for 2 consecutive):    │
  │   → STOP (converged — record result, move on)           │
  │                                                         │
  │ IF val_loss still decreasing > 0.5%/epoch:             │
  │   → CONTINUE to epoch 5 (still converging)              │
  │                                                         │
  │ ELSE: STOP and record                                   │
  └─────────────────────────────────────────────────────────┘

EPOCH 5 — ABSOLUTE MAXIMUM:
  ┌─────────────────────────────────────────────────────────┐
  │ UNCONDITIONAL STOP. Record final metrics.               │
  │ No trial may exceed 5 epochs under any circumstance.    │
  │                                                         │
  │ Reasoning: With aggressive pruning, most signal is      │
  │ captured by epoch 3-4. Epoch 5 catches slow-converging  │
  │ configs (high weight decay, small LR). Beyond 5 epochs  │
  │ the trial is either converged or overfitting.           │
  └─────────────────────────────────────────────────────────┘
```

### 18.3 Optuna Configuration for Aggressive Pruning

```python
import optuna
from optuna.pruners import MedianPruner, PercentilePruner

# AGGRESSIVE pruner configuration
pruner = MedianPruner(
    n_startup_trials=3,     # Only 3 unpruned trials before pruning starts
    n_warmup_steps=0,       # Prune from EPOCH 1 (no warmup!)
    interval_steps=1,       # Check every single epoch
    n_min_trials=3,         # Need 3 completed to establish median
)

# Alternative: PercentilePruner (even more aggressive)
# pruner = PercentilePruner(
#     percentile=60.0,       # Prune bottom 60% at each step
#     n_startup_trials=3,
#     n_warmup_steps=0,
#     interval_steps=1,
# )

study = optuna.create_study(
    direction="minimize",
    pruner=pruner,
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=5,    # Random exploration before TPE kicks in
        multivariate=True,     # Model parameter correlations
    ),
)

# In the objective function:
def objective(trial):
    # ... configure model with trial params ...
    
    MAX_EPOCHS = 5  # HARD CEILING
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader)
        val_loss = evaluate(model, val_loader)
        
        # Report to Optuna for pruning decision
        trial.report(val_loss, epoch)
        
        # ── EPOCH 1: Sanity Gate ──
        if epoch == 0:
            best_known = study.best_value if study.best_trial else float('inf')
            if val_loss > 3.0 * best_known:
                raise optuna.TrialPruned()
        
        # ── EPOCH 2: Competitive Gate ──
        if epoch == 1:
            if val_loss > history[0]:  # Loss INCREASED
                raise optuna.TrialPruned()
        
        # ── EPOCH 3: Hard Kill Zone ──
        if epoch == 2:
            improvement = (history[0] - val_loss) / history[0]
            if improvement < 0.05:  # < 5% total improvement
                raise optuna.TrialPruned()
        
        # ── MedianPruner also checks at every epoch ──
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # ── EPOCH 4: Convergence check ──
        if epoch == 3:
            recent_improvement = (history[-2] - val_loss) / history[-2]
            if recent_improvement < 0.001:  # Plateaued
                break  # Stop early, record result
        
        history.append(val_loss)
    
    return val_loss  # Best val_loss achieved in ≤5 epochs
```

### 18.4 Expected Trial Lifecycle Distribution

```
EXPECTED PRUNING DISTRIBUTION (per 30 trials):
═══════════════════════════════════════════════════════════════
Stage              Pruned    Survive    Cumulative Alive
═══════════════════════════════════════════════════════════════
Start              —         30         30 (100%)
After Epoch 1      6-8       22-24      22-24 (73-80%)
After Epoch 2      8-10      12-16      12-16 (40-53%)
After Epoch 3      6-10      4-8        4-8   (13-27%)
After Epoch 4      1-3       2-6        2-6   (7-20%)
Complete Epoch 5   —         2-6        2-6   (7-20%)
═══════════════════════════════════════════════════════════════

COMPUTE SAVINGS vs. 14-epoch trials:
  Old: 30 trials × 14 epochs average = 420 trial-epochs
  New: 30 trials × ~2.5 epochs average = ~75 trial-epochs
  → 5.6× FASTER Optuna search
  → Same GPU time runs 5× more trials → BETTER hyperparameter coverage

CRITICAL INSIGHT:
  With a 5-epoch max and aggressive pruning, you can run 
  150 trials in the same GPU-time that 30 trials at 14 epochs cost.
  More trials + aggressive pruning = BETTER results than fewer long trials.
```

---

## 19. PUBLIC DATA SOURCES, INSTITUTIONAL KNOWLEDGE & ARCHITECTURAL ADVANTAGES

### 19.1 ALL Publicly Available Data Sources

Every dataset below is freely accessible or has a free tier. Organized by institutional quality.

#### Tier 1 — Institutional-Grade Free Data

```
SOURCE                  TYPE                     ACCESS              LATENCY    COVERAGE
─────────────────────────────────────────────────────────────────────────────────────────
Federal Reserve (FRED)  Macro: rates, yield       fred.stlouisfed.org  Daily     1950-now
                        curve, inflation,         API key (free)
                        employment, GDP, M2

Yahoo Finance           OHLCV equities, crypto,   yfinance Python pkg  1-min     2010-now
                        options chains, splits,   Free, no key
                        dividends, fundamentals

Alpha Vantage           OHLCV, forex, crypto,     alphavantage.co      1-min     2000-now
                        fundamentals, econ        Free key (5/min)
                        indicators, SMA/EMA/RSI

Nasdaq Data Link        Corp fundamentals,        data.nasdaq.com      Daily     2000-now
(formerly Quandl)       economic indicators,      Free tier available
                        Zillow housing, futures

CBOE                    VIX, VIX term structure,  cboe.com/data        Daily     1990-now
                        put/call ratios, SKEW     Free CSV downloads

SEC EDGAR               13F filings (institutional sec.gov/edgar        Quarterly  1993-now
                        holdings), insider trades  Free, no key
                        (Form 4), 10-K/10-Q text

IEX Cloud               Real-time quotes, OHLCV,  iexcloud.io          Real-time  2015-now
                        fundamentals, news        Free tier (50K msg/mo)

Polygon.io              Stocks, options, forex,   polygon.io           Real-time  Varies
                        crypto aggregates         Free tier (5/min)

FINRA                   Short interest, ATS       finra.org/data       Bi-weekly  2010-now
                        (dark pool) volume        Free downloads

BLS (Bureau of Labor)   CPI, PPI, employment,     bls.gov              Monthly    1913-now
                        wages, productivity       Free API

Census Bureau           Retail sales, trade,      census.gov           Monthly    1992-now
                        housing starts, durables  Free API

ISM                     Manufacturing & services  ismworld.org         Monthly    1948-now
                        PMI (Purchasing Managers) Headline free

Treasury.gov            Yield curves, auction     treasury.gov         Daily      1990-now
                        results, debt outstanding Free CSV

CFTC                    Commitments of Traders    cftc.gov             Weekly     1986-now
                        (COT) — futures           Free CSV
                        positioning by type

CME Group               Futures settlement,       cmegroup.com         Daily      Varies
                        options volume, open      Free delayed data
                        interest by strike
```

#### Tier 2 — Alternative / Supplementary Free Data

```
SOURCE                  TYPE                     ACCESS              USE CASE
─────────────────────────────────────────────────────────────────────────
Binance API             Crypto OHLCV, order       api.binance.com     Crypto modeling
                        book snapshots, funding   Free, rate-limited
                        rates, liquidations

CoinGlass               Crypto funding rates,     coinglass.com       Crypto sentiment
                        open interest, long/short Free tier
                        ratios, liquidation data

GDELT Project           Global news events,       gdeltproject.org    Event-driven
                        sentiment, themes,        Free, massive
                        geographic tagging        (BigQuery)

Reddit (via Pushshift)  WallStreetBets, crypto    reddit API /        Social sentiment
                        subs, NLP-able text       pushshift

Twitter/X API           Financial tweets,         developer.x.com     Social momentum
                        ticker mentions           Free tier

Google Trends           Search interest for       trends.google.com   Retail attention
                        financial terms, tickers  Free API             proxy

Wikipedia Page Views    Views for company /       wikimedia.org       Attention signal
                        financial event pages     Free API

GitHub Stars/Activity   Open-source crypto        api.github.com      DeFi / tech proxy
                        project activity          Free

DataHub.io              Curated datasets: S&P     datahub.io          Reference data
                        500 list, country codes,  Free
                        currency codes
```

### 19.2 Institutional Finance Models & Quantitative Techniques

Models, methodologies, and frameworks used by top quantitative firms — all from published academic research and freely available papers.

#### Asset Pricing & Factor Models

```
MODEL                        REFERENCE                   FACTORS / COMPONENTS
──────────────────────────────────────────────────────────────────────────────
CAPM                         Sharpe (1964)               Market excess return (Rm - Rf)
Fama-French 3-Factor         Fama & French (1993)        Market + SMB (size) + HML (value)
Fama-French 5-Factor         Fama & French (2015)        + RMW (profitability) + CMA (investment)
Carhart 4-Factor             Carhart (1997)              FF3 + MOM (momentum)
Q-Factor Model               Hou, Xue, Zhang (2015)     Market + Size + Investment + ROE
Stambaugh-Yuan Mispricing    Stambaugh & Yuan (2017)     Management + Performance factors
Daniel-Hirshleifer-Sun       DHS (2020)                  Long-term + short-term overreaction

APPLICATION TO MK1:
  → Use Fama-French factor returns as ADDITIONAL FEATURES
  → Available free from Kenneth French's data library
  → URL: mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  → Daily factor returns: MKT-RF, SMB, HML, RMW, CMA, MOM
  → Add as 6 extra features to the existing 44 → 50 features
```

#### Volatility Models

```
MODEL              FORMULA / MECHANISM                    USE IN MK1
───────────────────────────────────────────────────────────────────────
GARCH(1,1)         σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁          Conditional volatility estimate →
                                                          add as feature, or use as
                                                          adaptive position sizing
                                                          
EGARCH             log(σ²ₜ) = ω + α·|εₜ₋₁/σₜ₋₁|         Captures asymmetric vol (leverage
                   + γ·(εₜ₋₁/σₜ₋₁) + β·log(σ²ₜ₋₁)       effect: drops increase vol more)
                   
GJR-GARCH          σ²ₜ = ω + (α+γ·I)·ε²ₜ₋₁ + β·σ²ₜ₋₁   Simpler asymmetric leverage model
                   I=1 if εₜ₋₁<0

HAR-RV             RV_d = β₀ + β_d·RV_d₋₁ +              Heterogeneous Autoregressive
(Corsi, 2009)      β_w·RV_w + β_m·RV_m                    Realized Volatility — captures
                   (daily + weekly + monthly)               multi-horizon volatility patterns
                   
Parkinson Vol      σ² = 1/(4·ln2) · (H-L)²               Already implemented as feature
(already in MK1)   Uses high-low range                     in indicators.py

Heston Model       dS = μS·dt + √v·S·dW₁                 Academic stochastic vol model
                   dv = κ(θ-v)·dt + ξ√v·dW₂              Use vol dynamics insight for
                   corr(W₁,W₂) = ρ                        feature engineering

APPLICATION TO MK1:
  → Add GARCH(1,1) conditional volatility as feature #45
  → Add HAR-RV (daily/weekly/monthly realized vol) as features #46-48
  → Use EGARCH leverage effect signal as feature #49
  → These are computable from existing OHLCV data — no new data source needed
```

#### Options / Derivatives Models

```
MODEL                  APPLICATION                           DATA SOURCE
──────────────────────────────────────────────────────────────────────────
Black-Scholes          Implied Volatility (IV) extraction    CBOE, Yahoo chains
                       → IV surface features for sentiment

Put-Call Ratio         Market fear gauge                     CBOE (free)
                       → Extreme readings predict reversals

VIX Term Structure     Contango/backwardation signal         CBOE (free)
                       → Contango = complacency
                       → Backwardation = fear

SKEW Index             Tail risk pricing by market           CBOE (free)
                       → High SKEW = crash protection demand

GEX (Gamma Exposure)   Dealer hedging → price pinning near   Derived from options OI
                       high-OI strikes (mechanical effect)

DIX (Dark Index)       Dark pool short volume ratio          squeezemetrics.com (free)
                       → Institutional positioning proxy

APPLICATION TO MK1:
  → Add VIX level + VIX 1m-3m term structure slope as features
  → Add put-call ratio as contrarian sentiment feature
  → Add SKEW index for tail risk awareness
  → ALL available daily from CBOE for free
```

#### Quantitative Trading Strategies (Published Academic Research)

```
STRATEGY              MECHANISM                    SEMINAL PAPER           YEARS TESTED
────────────────────────────────────────────────────────────────────────────────────────
MOMENTUM              Buy winners, sell losers     Jegadeesh & Titman      1965-1989
(cross-sectional)     12-month return, skip 1mo    (1993)                  Replicated globally

TIME-SERIES MOMENTUM  Go long when >0 return,      Moskowitz, Ooi,         1985-2009
(trend following)     short when <0 over lookback   Pedersen (2012)         200+ markets

MEAN REVERSION        Fade short-term extremes     Poterba & Summers       1926-1985
                      (1-week, Bollinger bands)    (1988)

STATISTICAL           Pairs: long cheap, short     Gatev, Goetzmann,       1962-2002
ARBITRAGE             expensive in corr pairs      Rouwenhorst (2006)

CARRY                 Buy high-yield, sell low-    Koijen et al. (2018)    1983-2012
                      yield (FX, bonds, futures)   "Carry" QJE             Global

VALUE (deep value)    Buy cheap assets (low P/E,   Asness, Moskowitz,      1972-2011
                      P/B) vs. expensive           Pedersen (2013)

LOW VOLATILITY        Low-vol stocks outperform    Baker, Bradley,         1968-2008
ANOMALY               (contradicts CAPM)           Wurgler (2011)

QUALITY               Buy high-quality (ROE,       Asness, Frazzini,       1957-2012
                      stable earnings, low debt)   Pedersen (2019)

LIQUIDITY PREMIUM     Illiquid assets earn premium Amihud & Mendelson      1964-1986
                      (trade less-liquid names)    (1986)

OPTION WRITING        Sell volatility (short puts  Various                 Ongoing
(short volatility)    or strangles) — harvests
                      variance risk premium

APPLICATION TO MK1:
  → Momentum features already partially captured (returns_1, returns_5,
    momentum_10, momentum_20 in indicators.py)
  → ADD: 12-month momentum (252 bars), skip-month momentum, sector momentum
  → ADD: Mean-reversion z-scores at multiple horizons
  → ADD: Volatility regime indicator (high-vol / low-vol state flag)
  → ADD: Carry signal (if applicable — yield curve slope from FRED)
```

### 19.3 Institutional Quantitative Firms — Published Research & Insights

Published, freely accessible papers and frameworks from leading quant firms:

```
FIRM                    PUBLISHED CONTRIBUTIONS                           ACCESS
──────────────────────────────────────────────────────────────────────────────────
AQR Capital             - "Value and Momentum Everywhere" (2013)          aqr.com/insights
Management              - "A Century of Evidence on Trend-Following"
(Cliff Asness)          - "Betting Against Beta", "Quality Minus Junk"
                        - Open-source factor data: AQR Data Library

Two Sigma               - "A Study of Carry" (2018)                      twosigma.com/insights
                        - Venn factor analytics platform (free)
                        - Halite reinforcement learning competition

Renaissance             - Publicly: Simons' Numberphile interview         No papers (secretive)
Technologies            - Key insight: find small, persistent edges
                        - Use non-financial PhD researchers
                        - Massive data cleaning investment

Citadel/                - Ken Griffin: macro + quant multi-strategy       Limited public
Citadel Securities      - Market-making technology insights
                        - Focus: execution quality, latency

DE Shaw                 - "Practical Considerations for ML in Finance"    deshaw.com/research
                        - David Shaw's molecular dynamics → finance
                        - Non-linear modeling research

Bridgewater             - "How the Economic Machine Works" (Dalio)        bridgewater.com
Associates              - "Principles for Navigating Big Debt Crises"
                        - All-Weather portfolio theory
                        - Risk parity methodology (free paper)

Man Group / AHL         - "Machine Learning for Factor Investing"         man.com/ahl/research
                        - Oxford-Man Institute of Quantitative Finance
                        - Realized volatility library (free)

WorldQuant               - "Finding Alphas" (book, open approach)         worldquant.com
                        - WorldQuant University (free MS in Finance)
                        - WebSim alpha discovery platform

BlackRock / Aladdin     - Factor investing research                      blackrock.com/corporate
                        - iShares factor ETF methodology papers           /literature
                        - Risk management frameworks

Quantopian (legacy)     - QuantCon presentations (YouTube)               quantopian.com (archive)
                        - Open-source: Zipline, Alphalens, Pyfolio
                        - Lecture series on algorithmic trading
```

### 19.4 Essential Quantitative Finance Libraries (Open Source)

```
LIBRARY              PURPOSE                              INSTALL
──────────────────────────────────────────────────────────────────────
zipline-reloaded     Event-driven backtesting engine       pip install zipline-reloaded
backtrader           Pythonic backtesting framework         pip install backtrader
vectorbt             Vectorized backtesting (fast)          pip install vectorbt
pyfolio              Performance & risk analytics           pip install pyfolio-reloaded
alphalens            Factor analysis (IC, quantile rets)    pip install alphalens-reloaded
empyrical            Risk metrics (Sharpe, VaR, drawdown)   pip install empyrical
arch                 GARCH/EGARCH/GJR-GARCH models          pip install arch
statsmodels          ADF test, ARIMA, cointegration          pip install statsmodels
hmmlearn             Hidden Markov Models (regime detect)    pip install hmmlearn
ruptures             Change point detection                  pip install ruptures
mlfinlab             ML for Finance (López de Prado)         pip install mlfinlab
ta-lib               130+ technical indicators               pip install TA-Lib
pandas-ta            130+ indicators (pure Python)           pip install pandas-ta
mamba-ssm            Mamba State Space Model                 pip install mamba-ssm
optuna               Hyperparameter optimization             (already installed)
shap                 Model explainability                    pip install shap
captum               PyTorch model interpretability          pip install captum
onnxruntime          Optimized model inference               pip install onnxruntime-gpu
```

### 19.5 Architectural Advantages — LNN + Mamba + XGBoost Hybrid

Why this specific architecture stack is superior to alternatives:

```
ARCHITECTURE COMPARISON:
═══════════════════════════════════════════════════════════════════════
Architecture          Strength           Weakness            For MK1?
═══════════════════════════════════════════════════════════════════════
Pure Transformer      Long-range deps    O(T²), data-hungry  ✗ (355K too small)
Pure LSTM/GRU         Sequential, light  Gradient vanishing   ✗ (long seqs)
Pure XGBoost          Tabular features   No sequence memory   ✗ (loses temporal)
Pure CNN              Multi-scale        No memory/state      ✗ (no persistence)
LNN (current)         Domain-adapted     Attention O(T²)      ~ (good, improvable)
Mamba SSM             O(n) linear        Newer, less tested   ✓ (best for regime)
LNN+Mamba+XGB (plan)  Best of all worlds Complexity           ✓✓✓ (RECOMMENDED)
═══════════════════════════════════════════════════════════════════════

WHY LNN + MAMBA + XGB IS THE OPTIMAL STACK:

1. MAMBA SSM (replaces Attention + GRU):
   - Processes sequences in O(n) linear time
   - Input-dependent state transitions = implicit regime detection
   - No KV cache, constant memory per token
   - Hardware-efficient parallel scan on GPU
   - Replaces both the quadratic attention AND the recurrent GRU
     with a single, faster, more expressive module

2. SQUEEZE-EXCITE GATING (retained):
   - Channel recalibration — learns which features matter when
   - Negligible compute overhead (32-dim bottleneck)
   - Feature-level attention complementary to temporal attention

3. XGBOOST RESIDUAL (retained):
   - Captures non-linear, non-sequential patterns Mamba misses
   - Decision tree ensembles excel at tabular feature interactions
   - Gradient boosting on residuals: learns what the neural net fails
   - Near-zero inference cost (0.01-0.05ms)
   - Acts as a "safety net" — even if Mamba degrades, XGB stabilizes

4. MC DROPOUT UNCERTAINTY (retained):
   - Position sizing proportional to model confidence
   - Risk management: skip low-confidence predictions
   - No additional parameters or training cost

5. INCEPTION/TIMESNET FEATURES (new):
   - Multi-scale learned features complement hand-coded indicators
   - Discovers patterns invisible to traditional TA
   - Concatenated (not replaced) — preserves domain knowledge

6. HYBRID ADVANTAGE:
   - Neural net: temporal patterns, regime adaptation, uncertainty
   - XGBoost: feature interactions, outlier robustness, interpretability
   - Together: error decorrelated → lower combined error than either alone

INFORMATION THEORY ARGUMENT:
  - Hand-coded indicators: ~3-4 bits of mutual info with target
  - Learned features (CNN): ~1-2 additional bits (orthogonal patterns)
  - Mamba temporal: ~2-3 bits (sequential structure)
  - XGBoost residual: ~0.5-1 bit (non-sequential interactions)
  - Total captured: ~7-10 bits — approaching theoretical maximum
    for OHLCV-only data at 1-min resolution
```

---

## SIGN-OFF

This plan represents a comprehensive roadmap to evolve ModelMK1 from its current retail-prototype state to institutional-grade production quality. The **142+ improvements** across **11 phases** address every known limitation in accuracy, robustness, efficiency, fail-safes, overfit/underfit prevention, and operational rigor. The addition of **Mamba SSM** (replacing MoE), **CPCV** (replacing walk-forward CV), **InceptionTime** (replacing basic Conv1D), and **aggressive Optuna pruning** (5-epoch max, 3-epoch kill zone) reflects current quantitative best practices.

**Critical Path to Meaningful Signal:**
1. Fix the target formulation (multi-task decoupling) — §3.2
2. Add Combinatorial Purged Cross-Validation (honest evaluation) — §6.1
3. Tune XGBoost jointly (end-to-end optimization) — §8.4
4. Add realistic backtesting (know true performance) — §9.1

Until these four items are complete, all other improvements build on an unreliable foundation.

**Success Criteria for Production Approval:**
- Directional accuracy > 54% consistently across CPCV combinations (statistically significant above random)
- Deflated Sharpe Ratio > 1.0 after costs
- CPCV Probability of Backtest Overfit (PBO) < 30%
- 30-day live shadow period with positive risk-adjusted return
- All circuit breakers implemented and tested
- Full pipeline inference < 100ms for 1-min bars, < 100ms for 5-min bars
- Model card and risk assessment documentation complete

---

*— Chief Architect, Model Development Division*

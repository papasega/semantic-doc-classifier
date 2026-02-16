# Mathematical & Theoretical Foundations

> Deep-dive into the mathematics behind semantic classification, vector embeddings, OCR, and the E5 model architecture. For usage instructions, see **[README.md](README.md)**.

---

## Table of Contents

1. [Vector Embeddings & Semantic Space](#1-vector-embeddings--semantic-space)
2. [Similarity Measures: Cosine vs Euclidean vs Dot Product](#2-similarity-measures-cosine-vs-euclidean-vs-dot-product)
3. [L2 Normalization & Metric Equivalence](#3-l2-normalization--metric-equivalence)
4. [Centroid-Based Classification](#4-centroid-based-classification)
5. [Confidence, Margins & Decision Boundaries](#5-confidence-margins--decision-boundaries)
6. [The Curse of Dimensionality](#6-the-curse-of-dimensionality)
7. [OCR Pipeline: Detection, Recognition & CTC](#7-ocr-pipeline-detection-recognition--ctc)
   - 7.1 [DBNet: Differentiable Binarization](#71-dbnet-differentiable-binarization)
   - 7.2 [CRNN + CTC Decoding](#72-crnn--ctc-decoding)
   - 7.3 [Angle Classification](#73-angle-classification)
   - 7.4 [Quality Metrics & Failure Modes](#74-quality-metrics--failure-modes)
   - 7.5 [PDF Extraction: Native vs OCR](#75-pdf-extraction-native-vs-ocr)
8. [E5-Large Multilingual: Architecture & Training](#8-e5-large-multilingual-architecture--training)
   - 8.1 [Contrastive Learning & InfoNCE Loss](#81-contrastive-learning--infonce-loss)
   - 8.2 [The Role of Prefixes](#82-the-role-of-prefixes)
   - 8.3 [Average Pooling vs CLS Token](#83-average-pooling-vs-cls-token)
   - 8.4 [Batch Processing & GPU Optimization](#84-batch-processing--gpu-optimization)
9. [Semantic Router: End-to-End Flow](#9-semantic-router-end-to-end-flow)

---

## 1. Vector Embeddings & Semantic Space

A text embedding model is a learned function:

```math
f_{\theta} : \mathcal{T} \rightarrow \mathbb{R}^d
```

It maps a text string from the space of all texts $\mathcal{T}$ to a point in a $d$-dimensional real vector space (here $d = 1024$), parameterized by model weights $\theta$.

The fundamental property of a well-trained embedding model is that **semantic similarity maps to geometric proximity**:

```math
\mathrm{sem}(t_1, t_2) \approx \mathrm{sim}\bigl(f_{\theta}(t_1),\; f_{\theta}(t_2)\bigr)
```

This means:

- Two invoices (one in French, one in English) → embeddings **close together**
- An invoice and a technical manual → embeddings **far apart**
- A "devis" (quote) → **closer to invoices** than to contracts, without any explicit rule

This geometric structure is what enables zero-fragility classification: we never match keywords, we measure distances in meaning-space.

---

## 2. Similarity Measures: Cosine vs Euclidean vs Dot Product

Given two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$, three natural measures exist:

### Dot Product (Inner Product)

```math
\langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i=1}^{d} a_i \cdot b_i
```

Measures both *directional alignment* and *magnitude*. Two vectors can have a high dot product simply because they are long. Computationally the fastest — a single BLAS Level 1 operation.

### Euclidean Distance (L2 Distance)

```math
d_2(\mathbf{a}, \mathbf{b}) = \lVert \mathbf{a} - \mathbf{b} \rVert_2 = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}
```

Measures straight-line distance. Sensitive to magnitude: two vectors pointing the same direction but with different lengths have non-zero distance.

### Cosine Similarity

```math
\mathrm{cossim}(\mathbf{a}, \mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\lVert \mathbf{a} \rVert_2 \cdot \lVert \mathbf{b} \rVert_2} = \frac{\displaystyle\sum_{i=1}^{d} a_i b_i}{\sqrt{\displaystyle\sum_{i=1}^{d} a_i^2} \;\cdot\; \sqrt{\displaystyle\sum_{i=1}^{d} b_i^2}}
```

Measures the *angle* θ between vectors, ignoring magnitudes entirely:

```math
\mathrm{cossim}(\mathbf{a}, \mathbf{b}) = \cos(\theta)
```

| Value | Geometric Meaning | Semantic Interpretation |
|-------|-------------------|------------------------|
| cos(θ) = 1 | Parallel (θ = 0°) | Identical semantics |
| cos(θ) = 0 | Orthogonal (θ = 90°) | Unrelated semantics |
| cos(θ) = -1 | Anti-parallel (θ = 180°) | Opposite semantics (rare) |

**Why cosine similarity wins for text:** A short and a long document about the same topic should be classified identically. Cosine similarity is magnitude-invariant — it captures semantic *direction*, not document length.

---

## 3. L2 Normalization & Metric Equivalence

A vector is **L2-normalized** (unit vector) when:

```math
\lVert \mathbf{v} \rVert_2 = \sqrt{\sum_{i=1}^{d} v_i^2} = 1
```

The normalization projects any vector onto the unit hypersphere:

```math
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert_2}
```

### The key insight: after normalization, all three metrics collapse

**Cosine similarity becomes a simple dot product:**

```math
\mathrm{cossim}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \frac{\hat{\mathbf{a}} \cdot \hat{\mathbf{b}}}{\underbrace{\lVert \hat{\mathbf{a}} \rVert_2}_{=1} \cdot \underbrace{\lVert \hat{\mathbf{b}} \rVert_2}_{=1}} = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}
```

**Euclidean distance becomes a monotonic function of cosine similarity:**

```math
\lVert \hat{\mathbf{a}} - \hat{\mathbf{b}} \rVert_2^2 = \lVert \hat{\mathbf{a}} \rVert_2^2 + \lVert \hat{\mathbf{b}} \rVert_2^2 - 2\langle\hat{\mathbf{a}}, \hat{\mathbf{b}}\rangle = 2 - 2\cos(\theta)
```

**Practical consequence:** With L2-normalized embeddings (our pipeline enforces `normalize_embeddings=True`):

- **max cosine similarity** ≡ **max dot product** ≡ **min Euclidean distance**
- We use the **dot product** because it is a single matrix multiplication on GPU:

```math
\mathbf{S} = \hat{\mathbf{Q}} \cdot \hat{\mathbf{C}}^{\top} \quad \in \mathbb{R}^{n \times K}
```

where $\hat{\mathbf{Q}} \in \mathbb{R}^{n \times d}$ (query matrix) and $\hat{\mathbf{C}} \in \mathbb{R}^{K \times d}$ (centroid matrix). This is a standard GEMM operation, hardware-optimized on CUDA via cuBLAS.

---

## 4. Centroid-Based Classification

Each document class $k \in \lbrace 1, \ldots, K \rbrace$ is defined by $m_k$ natural language descriptions (typically 4–8).

### Step 1 — Compute centroids (once at startup)

For each class k, embed all descriptions and take the mean:

```math
\mathbf{c}_k = \frac{1}{m_k} \sum_{j=1}^{m_k} f_{\theta}\bigl(s_j^{(k)}\bigr)
```

Then L2-normalize:

```math
\hat{\mathbf{c}}_k = \frac{\mathbf{c}_k}{\lVert \mathbf{c}_k \rVert_2}
```

The centroid represents the **semantic center of gravity** of the class. Averaging multiple descriptions provides robustness (no single phrasing dominates) and coverage (different facets of the concept).

### Step 2 — Classify a document

Embed the document text as a query, then compute dot products against all centroids:

```math
s_k = \hat{\mathbf{q}} \cdot \hat{\mathbf{c}}_k \quad \forall\; k \in \lbrace 1, \ldots, K \rbrace
```

The predicted class:

```math
\hat{k} = \underset{k}{\mathrm{arg\,max}} \; s_k
```

### Computational complexity

| Operation | Complexity | Time (A100) |
|-----------|-----------|-------------|
| Embedding (forward pass) | O(L² · d) | ~20ms |
| Similarity (dot product) | O(K · d) | <0.01ms |
| Sort top-k | O(K log K) | <0.001ms |
| **Total** | **Dominated by embedding** | **~20ms** |

For batch classification of n documents: a single O(n · K · d) matrix multiply. With K=7, d=1024, n=1000: ~7.2M FLOPs — under 1ms on GPU.

---

## 5. Confidence, Margins & Decision Boundaries

### Threshold-based rejection

The best score $s^{\*} = \max_k s_k$ is compared against a threshold τ (default: 0.40):

```math
\hat{k} = \begin{cases} \underset{k}{\mathrm{arg\,max}}\; s_k & \text{if } s^{*} \geq \tau \\ \texttt{UNKNOWN} & \text{if } s^{*} < \tau \end{cases}
```

### Classification margin

The margin between the top two candidates:

```math
\Delta = s^{(1)} - s^{(2)}
```

| Margin Δ | Interpretation | Recommended Action |
|----------|----------------|---------------------|
| Δ > 0.15 | Clear winner | High confidence |
| 0.05 < Δ ≤ 0.15 | Moderate | Valid, worth monitoring |
| Δ ≤ 0.05 | Near-tie | Consider human review |

### Voronoi tessellation

Each centroid defines a **Voronoi cell** — all points closer to it than to any other centroid:

```math
V_k = \bigl\lbrace\; \mathbf{x} \in \mathcal{S}^{d-1} \;\bigm|\; \hat{\mathbf{x}} \cdot \hat{\mathbf{c}}_k \geq \hat{\mathbf{x}} \cdot \hat{\mathbf{c}}_j \;\; \forall\, j \neq k \;\bigr\rbrace
```

The **decision boundary** between classes i and j is the hyperplane:

```math
H_{ij} = \bigl\lbrace\; \mathbf{x} \in \mathbb{R}^d \;\bigm|\; \mathbf{x} \cdot (\hat{\mathbf{c}}_i - \hat{\mathbf{c}}_j) = 0 \;\bigr\rbrace
```

Documents near this hyperplane have small margins — genuine semantic ambiguity.

---

## 6. The Curse of Dimensionality

In high-dimensional spaces (d > 100), all pairwise distances concentrate around their mean:

```math
\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0
```

This does **not** break our classifier because:

1. **Embeddings are not uniform** — they cluster by topic, creating exploitable local structure
2. **We compare against K=7 centroids**, not millions of points — averaging already reduces noise
3. **Cosine similarity on normalized vectors** measures angular separation, which is more robust than absolute distance in high dimensions

The curse of dimensionality becomes critical for ANN search in vector databases (Step 2: RAG with HNSW) — but not for classification against a small set of centroids.

---

## 7. OCR Pipeline: Detection, Recognition & CTC

PaddleOCR implements a three-stage cascade:

```
Image → Detection (DBNet) → Angle Classification → Recognition (CRNN+CTC) → Text
```

### 7.1 DBNet: Differentiable Binarization

**Problem:** Given image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, detect text regions as bounding polygons.

**Architecture:** ResNet backbone → FPN (multi-scale fusion) → probability map P + threshold map T.

Traditional binarization is a hard threshold (non-differentiable):

```math
B_{i,j} = \begin{cases} 1 & \text{if } P_{i,j} \geq t \\ 0 & \text{otherwise} \end{cases}
```

DBNet replaces it with a differentiable approximation:

```math
\hat{B}_{i,j} = \frac{1}{1 + e^{-k \cdot (P_{i,j} - T_{i,j})}}
```

- $P_{i,j}$: text probability at pixel (i,j)
- $T_{i,j}$: adaptive threshold, **learned** by the network
- $k$: amplification factor (typically 50) — controls sigmoid sharpness

As $k \to \infty$, this sigmoid → hard step function. During training, k stays finite for gradient flow.

**Box confidence:**

```math
\mathrm{score} = \frac{1}{|R|} \sum_{(i,j) \in R} P_{i,j}
```

The `det_db_box_thresh` parameter (0.5) filters out low-confidence detections.

### 7.2 CRNN + CTC Decoding

Each detected text region → cropped → fed to the recognizer.

**Architecture:**

1. **CNN** (ResNet/MobileNet): image → feature sequence $\mathbf{F} \in \mathbb{R}^{T \times d_f}$
2. **BiLSTM**: models character dependencies

```math
\overrightarrow{\mathbf{h}}_t = \mathrm{LSTM_{fwd}}(\mathbf{F}_t,\; \overrightarrow{\mathbf{h}}_{t-1})
```

```math
\overleftarrow{\mathbf{h}}_t = \mathrm{LSTM_{bwd}}(\mathbf{F}_t,\; \overleftarrow{\mathbf{h}}_{t+1})
```

```math
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \;;\; \overleftarrow{\mathbf{h}}_t]
```

3. **Softmax over characters** at each time step:

```math
P(\pi_t = c \mid \mathbf{x}) = \mathrm{softmax}(\mathbf{W}\mathbf{h}_t + \mathbf{b})_c
```

**CTC decoding** solves the alignment problem (T time steps → L characters, L ≤ T). It introduces a blank token ε and a many-to-one mapping that removes blanks and collapses repeats:

```
B(-HH-EE-LL-LO-) = "HELLO"
```

The CTC loss marginalizes over all valid alignments:

```math
P(\mathbf{y} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} P(\pi_t \mid \mathbf{x})
```

**Per-line confidence** is the geometric mean along the best CTC path:

```math
\mathrm{conf} = \left(\prod_{t=1}^{T} P(\pi_t^{*} \mid \mathbf{x})\right)^{1/T}
```

### 7.3 Angle Classification

A lightweight MobileNet-v3 classifier detects 180°-rotated text:

```math
P(\text{rotation} = 180° \mid \text{crop}) \in [0, 1]
```

If P > 0.5, the crop is flipped before recognition. Critical for scans fed upside-down. Adds ~1ms per region.

### 7.4 Quality Metrics & Failure Modes

**Document-level quality:**

```math
\bar{c} = \frac{1}{n} \sum_{i=1}^{n} c_i
```

| Threshold | Interpretation | Action |
|-----------|---------------|--------|
| c̄ ≥ 0.85 | High quality | Proceed |
| 0.65 ≤ c̄ < 0.85 | Acceptable | Log info |
| c̄ < 0.65 | Low quality | Warning, flag for review |
| c̄ = 0 | No text | Returns UNKNOWN via threshold |

**Common failures:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| c̄ < 0.5, garbled | Degraded scan | ↑ DPI to 300+ |
| Empty result | Not a text image | Expected behavior |
| Wrong characters | Wrong `ocr_lang` | Set correct language |
| Missing regions | Threshold too high | Lower `det_db_box_thresh` |

### 7.5 PDF Extraction: Native vs OCR

Dual strategy per page:

- **Native text > 50 chars** → use PyMuPDF extraction directly (fast, perfect quality)
- **Native text ≤ 50 chars** → rasterize at 300 DPI → PaddleOCR

| DPI | A4 Resolution | Quality | Speed |
|-----|--------------|---------|-------|
| 150 | 1240 × 1754 | Marginal | ~150ms |
| **300** | **2480 × 3508** | **Excellent** | **~400ms** |
| 600 | 4960 × 7016 | Diminishing returns | ~1200ms |

---

## 8. E5-Large Multilingual: Architecture & Training

`intfloat/multilingual-e5-large`: 560M params, XLM-RoBERTa-Large base, 1024-dim embeddings, 100+ languages, 514 max tokens.

### 8.1 Contrastive Learning & InfoNCE Loss

Given query q, positive doc d⁺, and N-1 in-batch negatives:

```math
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\bigl(\mathrm{sim}(q, d^{+}) / \tau\bigr)}{\exp\bigl(\mathrm{sim}(q, d^{+}) / \tau\bigr) + \displaystyle\sum_{j=1}^{N-1} \exp\bigl(\mathrm{sim}(q, d_j^{-}) / \tau\bigr)}
```

- sim(a,b) = cosine similarity
- τ ≈ 0.01–0.05 (temperature — lower = peakier distribution)

This is an N-way softmax that identifies the positive among N candidates. With batch size B, in-batch negatives give O(B²) training signal per batch.

**Three-stage training:**
1. Weak supervision on ~1B web text pairs
2. Supervised fine-tuning (MS MARCO, NLI, multilingual corpora)
3. Instruction tuning for prefix-based behavior

### 8.2 The Role of Prefixes

| Use Case | Prefix | Strategy |
|----------|--------|----------|
| Indexing (documents) | `passage: ` | Emphasizes information content |
| Searching / classifying | `query: ` | Emphasizes intent |

The prefix is a **conditional signal** that activates different encoding strategies learned during training. Omitting it degrades nDCG@10 by 5–15%.

**In our pipeline:**
- Route descriptions → `passage:` (reference documents)
- Document to classify → `query:` (searching for the matching class)

### 8.3 Average Pooling vs CLS Token

The transformer outputs hidden states $\mathbf{H} \in \mathbb{R}^{L \times d}$. We convert to a single vector via **masked average pooling**:

```math
\mathbf{v} = \frac{\sum_{i=1}^{L} m_i \cdot \mathbf{h}_i}{\sum_{i=1}^{L} m_i}
```

where $m_i \in \lbrace 0, 1 \rbrace$ is the attention mask.

Average pooling outperforms [CLS]-based pooling because it aggregates from all positions, is more robust to positional biases, and has lower variance across runs.

### 8.4 Batch Processing & GPU Optimization

```
N texts → Tokenize + Pad → Tensor (N, L) → GPU forward → (N, L, 1024) → Pool → (N, 1024)
```

**VRAM estimation:**

```math
\text{VRAM} \approx \text{model} + B \times L \times d \times 4
```

For E5-large (B=32, L=512): ~4.5GB model + ~0.2GB activations ≈ 4.7GB.

**Throughput:**

| GPU | batch=16 | batch=32 | batch=64 |
|-----|----------|----------|----------|
| T4 (16GB) | ~150/s | ~200/s | OOM |
| A10G (24GB) | ~250/s | ~350/s | ~400/s |
| A100 (40GB) | ~350/s | ~500/s | ~650/s |

---

## 9. Semantic Router: End-to-End Flow

```
                  INIT (once)                              INFERENCE (per doc)
    ┌──────────────────────────────┐       ┌────────────────────────────────────┐
    │                              │       │                                    │
    │  Route descriptions          │       │  Document text                     │
    │  (4-8 per class, K classes) │       │  "FACTURE N°2024                  │
    │                              │       │   Total TTC: 15300€"              │
    │         │                    │       │          │                         │
    │   embed_documents()          │       │    embed_query()                   │
    │   prefix: "passage:"        │       │    prefix: "query:"               │
    │         │                    │       │          │                         │
    │   mean + L2 normalize        │       │    ┌─────▼──────┐                  │
    │         │                    │       │    │ query_vec   │                  │
    │   ┌─────▼──────┐             │       │    │  (1024,)    │                  │
    │   │ centroid_k  │────────────│───────│──> │             │                  │
    │   │  (1024,)    │            │       │    │ q @ C.T     │                  │
    │   └────────────┘             │       │    │ = scores(K) │                  │
    │                              │       │    └─────┬───────┘                  │
    │   C: (K, 1024) in memory     │       │    argmax + τ                      │
    └──────────────────────────────┘       │          │                         │
                                           │    FACTURE (0.847)                 │
                                           └────────────────────────────────────┘
```

| Phase | Complexity | Typical Time |
|-------|-----------|-------------|
| Init: embed M descriptions | O(M · L² · d) | ~2s (once) |
| Init: compute K centroids | O(K · m · d) | <1ms |
| Classify 1 document | O(L² · d + K · d) | ~25ms |
| Classify n docs (batch) | O(n · L² · d + n · K · d) | ~200ms (n=100) |

---

*For implementation details and usage, see [README.md](README.md).*

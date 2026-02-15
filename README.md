# Document Intelligence Pipeline — Semantic Classification

**Production-grade document classification using vector embeddings and cosine similarity.**
**Zero regex. Zero keyword matching. Pure vector geometry.**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Principle: Zero-Fragility Classification](#2-core-principle-zero-fragility-classification)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - 3.1 [Vector Embeddings & Semantic Space](#31-vector-embeddings--semantic-space)
   - 3.2 [Cosine Similarity vs Euclidean Distance vs Dot Product](#32-cosine-similarity-vs-euclidean-distance-vs-dot-product)
   - 3.3 [Why Normalized Vectors Simplify Everything](#33-why-normalized-vectors-simplify-everything)
   - 3.4 [Centroid-Based Classification](#34-centroid-based-classification)
   - 3.5 [Confidence Thresholding & Decision Boundaries](#35-confidence-thresholding--decision-boundaries)
   - 3.6 [The Curse of Dimensionality](#36-the-curse-of-dimensionality)
4. [OCR Theory & Pipeline](#4-ocr-theory--pipeline)
   - 4.1 [The Three-Stage Cascade](#41-the-three-stage-cascade)
   - 4.2 [Text Detection: DBNet Architecture](#42-text-detection-dbnet-architecture)
   - 4.3 [Text Recognition: CRNN + CTC Decoding](#43-text-recognition-crnn--ctc-decoding)
   - 4.4 [Angle Classification for Rotated Scans](#44-angle-classification-for-rotated-scans)
   - 4.5 [Quality Metrics & Failure Modes](#45-quality-metrics--failure-modes)
   - 4.6 [PDF Extraction Strategy: Native vs OCR](#46-pdf-extraction-strategy-native-vs-ocr)
5. [Embedding Engine: E5-Large Multilingual](#5-embedding-engine-e5-large-multilingual)
   - 5.1 [Architecture & Training Objective](#51-architecture--training-objective)
   - 5.2 [Contrastive Learning: InfoNCE Loss](#52-contrastive-learning-infonce-loss)
   - 5.3 [The Critical Role of Prefixes](#53-the-critical-role-of-prefixes)
   - 5.4 [L2 Normalization & Unit Hypersphere](#54-l2-normalization--unit-hypersphere)
   - 5.5 [Average Pooling: From Tokens to Sentences](#55-average-pooling-from-tokens-to-sentences)
   - 5.6 [Batch Processing & GPU Optimization](#56-batch-processing--gpu-optimization)
6. [Semantic Router: How Classification Works](#6-semantic-router-how-classification-works)
7. [Project Structure](#7-project-structure)
8. [Quick Start](#8-quick-start)
9. [API Reference](#9-api-reference)
10. [Configuration](#10-configuration)
11. [Adding New Document Classes](#11-adding-new-document-classes)
12. [Testing](#12-testing)
13. [Deployment](#13-deployment)
14. [Performance Benchmarks](#14-performance-benchmarks)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CLASSIFICATION PIPELINE                            │
│                                                                         │
│  ┌──────────┐    ┌────────────┐    ┌───────────────┐    ┌────────────┐ │
│  │          │    │            │    │               │    │            │ │
│  │  FILE    │───>│   TEXT     │───>│  EMBEDDING    │───>│  SEMANTIC  │ │
│  │  INPUT   │    │  EXTRACT   │    │  ENGINE       │    │  ROUTER    │ │
│  │          │    │            │    │               │    │            │ │
│  └──────────┘    └────────────┘    └───────────────┘    └────────────┘ │
│   PDF/DOCX/       PaddleOCR or     E5-large              Cosine       │
│   IMG/TXT          Native Text     1024-dim               Similarity   │
│                    Extraction      L2-normalized          vs Centroids │
│                                                                         │
│  Output: ClassificationResult{type, confidence, top_k, latency}        │
└─────────────────────────────────────────────────────────────────────────┘
```

The pipeline is split into three independent, composable stages:

**Stage 1 — Text Extraction:** Detects the file format and extracts raw text. For native PDFs and DOCX files, text is extracted directly. For scanned documents and images, PaddleOCR performs detection, angle correction, and recognition.

**Stage 2 — Embedding:** The extracted text is projected into a 1024-dimensional vector space using `intfloat/multilingual-e5-large`. The resulting vector captures the *semantic meaning* of the document, not its surface-level keywords.

**Stage 3 — Classification:** The document embedding is compared against pre-computed class centroids using cosine similarity. The class with the highest similarity score wins, subject to a confidence threshold.

---

## 2. Core Principle: Zero-Fragility Classification

Traditional document classifiers rely on brittle rule chains:

```python
# ❌ FRAGILE — What this project eliminates
if "facture" in text or "invoice" in text:
    return "FACTURE"
elif "contrat" in text and "article" in text:
    return "CONTRAT"
elif any(kw in text for kw in ["rapport", "analyse", "recommandation"]):
    return "RAPPORT"
else:
    return "UNKNOWN"
```

**Why this fails:**

- **Synonyms:** "note de débit" is an invoice but contains neither "facture" nor "invoice"
- **Multilingual content:** A contract in English won't match `"contrat" in text`
- **OCR errors:** "factur3" or "f@cture" from a degraded scan won't match
- **New vocabulary:** Every new term requires a new rule
- **Combinatorial explosion:** Each class × each language × each synonym = unmaintainable

**Our approach: no rules, no keywords, no regex.** Classification is performed entirely in vector space:

```python
# ✅ ZERO-FRAGILITY — What this project implements
similarities = query_embedding @ centroids.T   # Pure linear algebra
predicted_class = argmax(similarities)          # Geometric decision
```

The system "understands" that a "devis" (quote) is semantically close to a "facture" (invoice) because their embeddings are geometrically proximate in 1024-dimensional space — without anyone writing a rule for it.

---

## 3. Mathematical Foundations

### 3.1 Vector Embeddings & Semantic Space

A text embedding model is a learned function:

$$
f_\theta : \mathcal{T} \rightarrow \mathbb{R}^d
$$

that maps a text string from the space of all texts $\mathcal{T}$ to a point in a $d$-dimensional real vector space (here $d = 1024$), parameterized by model weights $\theta$.

The key property of a *well-trained* embedding model is that **semantic similarity in natural language maps to geometric proximity in vector space**:

$$
\text{sem}(t_1, t_2) \approx \text{sim}\big(f_\theta(t_1),\; f_\theta(t_2)\big)
$$

where $\text{sem}$ is an abstract notion of semantic similarity between texts $t_1$ and $t_2$, and $\text{sim}$ is a geometric similarity measure between their embeddings.

Concretely, this means:

- Two invoices (one in French, one in English) will have embeddings that are **close together**
- An invoice and a technical manual will have embeddings that are **far apart**
- A "devis" (quote) will be **closer to invoices** than to contracts, even though the word "devis" never appears in the invoice training data

This geometric structure is what enables zero-fragility classification.

### 3.2 Cosine Similarity vs Euclidean Distance vs Dot Product

Given two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$, three natural measures of similarity exist:

**Dot Product (Inner Product):**

$$
\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i \cdot b_i
$$

This measures both *directional alignment* and *magnitude*. Two vectors can have a high dot product simply because they are long, not because they point in the same direction. Computationally, this is a single `BLAS Level 1` operation — the fastest of the three.

**Euclidean Distance (L2 Distance):**

$$
d_2(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}
$$

This measures the "straight line" distance between two points in the vector space. Smaller distance = more similar. It is sensitive to vector magnitude: two vectors pointing in the same direction but with different lengths will have a non-zero distance.

**Cosine Similarity:**

$$
\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\|\mathbf{a}\|_2 \cdot \|\mathbf{b}\|_2} = \frac{\displaystyle\sum_{i=1}^{d} a_i \cdot b_i}{\sqrt{\displaystyle\sum_{i=1}^{d} a_i^2} \;\cdot\; \sqrt{\displaystyle\sum_{i=1}^{d} b_i^2}}
$$

This measures the *angle* $\theta$ between two vectors, completely ignoring their magnitudes:

$$
\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \cos(\theta)
$$

| Value | Geometric Meaning | Semantic Interpretation |
|-------|-------------------|------------------------|
| $\cos(\theta) = 1$ | Vectors are parallel ($\theta = 0°$) | Identical semantics |
| $\cos(\theta) = 0$ | Vectors are orthogonal ($\theta = 90°$) | Unrelated semantics |
| $\cos(\theta) = -1$ | Vectors are anti-parallel ($\theta = 180°$) | Opposite semantics (rare in practice) |

**Why cosine similarity is preferred for text classification:** A short document and a long document about the same topic should be classified identically. Cosine similarity is **magnitude-invariant** — it depends only on the *direction* of the embedding, which captures semantic content rather than document length.

### 3.3 Why Normalized Vectors Simplify Everything

A vector $\mathbf{v}$ is **L2-normalized** (unit vector) when:

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} v_i^2} = 1
$$

The normalization operation projects any non-zero vector onto the unit hypersphere:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}
$$

**After L2 normalization, all three metrics become equivalent:**

*Cosine similarity becomes a dot product:*

$$
\text{cos\_sim}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \frac{\hat{\mathbf{a}} \cdot \hat{\mathbf{b}}}{\underbrace{\|\hat{\mathbf{a}}\|_2}_{=1} \cdot \underbrace{\|\hat{\mathbf{b}}\|_2}_{=1}} = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}
$$

*Euclidean distance becomes a monotonic function of cosine similarity:*

$$
\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = \|\hat{\mathbf{a}}\|_2^2 + \|\hat{\mathbf{b}}\|_2^2 - 2\langle\hat{\mathbf{a}}, \hat{\mathbf{b}}\rangle = 2 - 2\cos(\theta)
$$

**Practical consequence:** With L2-normalized embeddings (enforced by `normalize_embeddings=True` in our pipeline):

- **Maximizing cosine similarity** $\equiv$ **maximizing dot product** $\equiv$ **minimizing Euclidean distance**
- We use the **dot product** because it is the fastest: a single matrix multiplication on GPU

For a batch of $n$ documents against $K$ class centroids:

$$
\mathbf{S} = \hat{\mathbf{Q}} \cdot \hat{\mathbf{C}}^T \quad \in \mathbb{R}^{n \times K}
$$

where $\hat{\mathbf{Q}} \in \mathbb{R}^{n \times d}$ is the query matrix and $\hat{\mathbf{C}} \in \mathbb{R}^{K \times d}$ is the centroid matrix. This is a standard `GEMM` (General Matrix Multiply) operation, hardware-optimized on CUDA via cuBLAS. On an A100, this completes in microseconds for typical workloads.

### 3.4 Centroid-Based Classification

Each document class $k \in \{1, \ldots, K\}$ is defined by a set of $m_k$ natural language descriptions $\{s_1^{(k)}, s_2^{(k)}, \ldots, s_{m_k}^{(k)}\}$ (typically 4–8 descriptions per class).

**Step 1 — Compute class centroids (done once at startup):**

For each class $k$, embed all its descriptions using the embedding model $f_\theta$ and compute the arithmetic mean:

$$
\mathbf{c}_k = \frac{1}{m_k} \sum_{j=1}^{m_k} f_\theta(s_j^{(k)})
$$

Then L2-normalize the centroid to project it onto the unit hypersphere:

$$
\hat{\mathbf{c}}_k = \frac{\mathbf{c}_k}{\|\mathbf{c}_k\|_2}
$$

The centroid $\hat{\mathbf{c}}_k$ represents the **"semantic center of gravity"** of class $k$ in the embedding space. By averaging multiple descriptions, we achieve two properties:

1. **Robustness:** The centroid is not sensitive to any single phrasing or word choice
2. **Coverage:** Different descriptions capture different facets of the concept (formal/informal, FR/EN, synonyms)

**Step 2 — Classify a document:**

Given a document with text $t$, compute its embedding $\hat{\mathbf{q}} = \text{normalize}(f_\theta(t))$. Then compute similarities against all centroids:

$$
s_k = \hat{\mathbf{q}} \cdot \hat{\mathbf{c}}_k \quad \forall k \in \{1, \ldots, K\}
$$

The predicted class is:

$$
\hat{k} = \underset{k \in \{1,\ldots,K\}}{\arg\max} \; s_k
$$

**Computational complexity per classification:**

| Operation | Complexity | Time (A100) |
|-----------|-----------|-------------|
| Embedding (forward pass) | $O(L^2 \cdot d_{\text{model}})$ | ~20ms |
| Similarity computation | $O(K \cdot d)$ | <0.01ms |
| Sorting top-k | $O(K \log K)$ | <0.001ms |
| **Total** | **Dominated by embedding** | **~20ms** |

Where $L$ is the sequence length (tokens) and $d_{\text{model}}$ is the transformer hidden dimension.

### 3.5 Confidence Thresholding & Decision Boundaries

The raw similarity score $s^* = \max_k s_k$ measures how well the document matches the best class. We apply a threshold $\tau$ (default: 0.40):

$$
\hat{k} = \begin{cases} \underset{k}{\arg\max}\; s_k & \text{if } s^* \geq \tau \\ \texttt{UNKNOWN} & \text{if } s^* < \tau \end{cases}
$$

**The margin** between the top two candidates provides additional signal:

$$
\Delta = s^{(1)} - s^{(2)}
$$

where $s^{(1)}$ and $s^{(2)}$ are the highest and second-highest scores respectively.

| Margin $\Delta$ | Interpretation | Action |
|----------------|----------------|--------|
| $\Delta > 0.15$ | Clear winner | High confidence classification |
| $0.05 < \Delta \leq 0.15$ | Moderate separation | Classification valid but worth monitoring |
| $\Delta \leq 0.05$ | Near-tie | Consider human review |

**Geometric interpretation — Voronoi tessellation:**

In the embedding space, each centroid $\hat{\mathbf{c}}_k$ defines a **Voronoi cell** — the set of all points closer to $\hat{\mathbf{c}}_k$ than to any other centroid:

$$
V_k = \big\{\ \mathbf{x} \in \mathcal{S}^{d-1} \ \big|\ \hat{\mathbf{x}} \cdot \hat{\mathbf{c}}_k \geq \hat{\mathbf{x}} \cdot \hat{\mathbf{c}}_j \;\; \forall j \neq k \ \big\}
$$

The **decision boundary** between two classes $i$ and $j$ is the hyperplane:

$$
H_{ij} = \big\{\ \mathbf{x} \in \mathbb{R}^d \ \big|\ \mathbf{x} \cdot (\hat{\mathbf{c}}_i - \hat{\mathbf{c}}_j) = 0 \ \big\}
$$

Documents near this hyperplane will have small margins $\Delta$, indicating genuine semantic ambiguity between the two classes.

### 3.6 The Curse of Dimensionality

In high-dimensional spaces ($d > 100$), a counter-intuitive phenomenon occurs: **all pairwise distances concentrate around their mean**. For $n$ points uniformly distributed in the unit cube $[0,1]^d$:

$$
\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0
$$

This means the contrast between the nearest and farthest neighbor vanishes. However, this does **not** break our classifier because:

1. **Embeddings are not uniformly distributed.** They cluster by semantic topic, creating exploitable local structure.
2. **We compare against $K = 7$ centroids, not millions of points.** The centroid computation already averages out noise.
3. **Cosine similarity on normalized vectors** is more robust than Euclidean distance in high dimensions because it measures angular separation rather than absolute distance.

The curse of dimensionality becomes critical when scaling to approximate nearest neighbor search in vector databases — but that is a concern for Step 2 (RAG), not for the classification step.

---

## 4. OCR Theory & Pipeline

### 4.1 The Three-Stage Cascade

PaddleOCR implements a three-stage cascade architecture:

```
                    ┌─────────────┐
                    │   INPUT     │
                    │   IMAGE     │
                    │ (H × W × 3)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  DETECTION  │
                    │   (DBNet)   │     Stage 1: WHERE is the text?
                    │             │     Output: Bounding polygons
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ANGLE     │
                    │   CLASS.    │     Stage 2: Is the text rotated 180°?
                    │             │     Output: Rotation correction
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ RECOGNITION │
                    │ (CRNN+CTC)  │     Stage 3: WHAT does the text say?
                    │             │     Output: Text + confidence
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   RESULT    │
                    │ text, conf, │
                    │   bbox      │
                    └─────────────┘
```

Each stage is a separate neural network, running sequentially on the GPU. This modular design allows each component to be optimized independently.

### 4.2 Text Detection: DBNet Architecture

**Problem formulation:** Given an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, produce a set of bounding polygons $\{P_1, P_2, \ldots, P_n\}$ where each $P_i$ encloses a text region.

**Architecture:** DBNet (Differentiable Binarization Network) uses:

1. **ResNet-18/50 backbone** — extracts multi-scale feature maps $\{F_2, F_3, F_4, F_5\}$ at strides $\{4, 8, 16, 32\}$
2. **Feature Pyramid Network (FPN)** — fuses features across scales to detect text of varying sizes
3. **Prediction heads** — output a probability map $P \in [0,1]^{H \times W}$ and a threshold map $T \in [0,1]^{H \times W}$

**The key innovation — Differentiable Binarization:**

Traditional binarization applies a hard threshold to convert the probability map to a binary mask:

$$
B_{i,j} = \begin{cases} 1 & \text{if } P_{i,j} \geq t \\ 0 & \text{otherwise} \end{cases}
$$

This step function has zero gradient almost everywhere, making it impossible to train end-to-end via backpropagation. DBNet replaces it with an approximate, differentiable step function:

$$
\hat{B}_{i,j} = \frac{1}{1 + e^{-k \cdot (P_{i,j} - T_{i,j})}}
$$

where:
- $P_{i,j}$ is the probability of text at pixel $(i,j)$
- $T_{i,j}$ is the **adaptive threshold** at pixel $(i,j)$, learned by the network
- $k$ is the amplification factor (typically $k = 50$), controlling the sharpness

As $k \to \infty$, this sigmoid approaches the hard step function. During training, $k$ is kept finite for gradient flow; during inference, the hard threshold can be used.

**The `det_db_box_thresh` parameter** (0.5 in our config) sets the minimum average probability inside a detected box for it to be kept. This is a precision/recall trade-off:

$$
\text{box\_score} = \frac{1}{|R|} \sum_{(i,j) \in R} P_{i,j}
$$

where $R$ is the set of pixels inside the detected polygon.

### 4.3 Text Recognition: CRNN + CTC Decoding

Once text regions are detected and cropped, each region is fed to the recognition network.

**Architecture:** The CRNN (Convolutional Recurrent Neural Network) has three components:

1. **Convolutional layers** (ResNet/MobileNet backbone): Extract visual feature maps from the cropped text image $\mathbf{x} \in \mathbb{R}^{h \times w \times 3}$, producing a feature sequence $\mathbf{F} \in \mathbb{R}^{T \times d_f}$ where $T$ is the number of horizontal positions and $d_f$ is the feature dimension.

2. **Bidirectional LSTM**: Models sequential dependencies between characters. For each time step $t$:

$$
\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{F}_t, \overrightarrow{\mathbf{h}}_{t-1})
\quad\quad
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{F}_t, \overleftarrow{\mathbf{h}}_{t+1})
$$

$$
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \;;\; \overleftarrow{\mathbf{h}}_t]
$$

3. **Linear + Softmax**: Produces a probability distribution over the character set $\mathcal{C} \cup \{\text{blank}\}$ at each time step:

$$
P(\pi_t = c \;|\; \mathbf{x}) = \text{softmax}(\mathbf{W} \mathbf{h}_t + \mathbf{b})_c
$$

**CTC (Connectionist Temporal Classification) Decoding:**

CTC solves the **alignment problem**: the feature sequence has $T$ time steps, but the output text has $L$ characters ($L \leq T$). There is no explicit alignment between positions and characters.

CTC introduces a *blank* token $\epsilon$ and defines a many-to-one mapping $\mathcal{B}$:

$$
\mathcal{B}(\pi_1, \pi_2, \ldots, \pi_T) = \text{text}
$$

by (1) removing repeated characters and (2) removing blanks. For example:

$$
\mathcal{B}(\text{-HH-EE-LL-LO-}) = \text{"HELLO"}
$$

The CTC loss marginalizes over all valid alignments $\pi$ that produce the target text $\mathbf{y}$:

$$
P(\mathbf{y} \;|\; \mathbf{x}) = \sum_{\pi \;\in\; \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} P(\pi_t \;|\; \mathbf{x})
$$

At inference, the most likely text is found via beam search or greedy decoding.

**The confidence score** for each recognized line is derived from the product of per-character probabilities along the best CTC path:

$$
\text{conf} = \left(\prod_{t=1}^{T} P(\pi_t^* \;|\; \mathbf{x})\right)^{1/T}
$$

where $\pi^*$ is the best CTC path (geometric mean for numerical stability).

### 4.4 Angle Classification for Rotated Scans

The `use_angle_cls=True` setting enables a lightweight CNN classifier that detects whether a text region is rotated 180°. This is critical for scanned documents fed into the scanner upside-down.

**Architecture:** A small MobileNet-v3 classifier that takes a cropped text region and outputs:

$$
P(\text{rotation} = 180° \;|\; \text{crop}) \in [0, 1]
$$

If $P > 0.5$, the crop is rotated 180° before recognition. This adds ~1ms per text region.

### 4.5 Quality Metrics & Failure Modes

Our OCR engine reports quality at two granularities:

**Per-line confidence** $c_i$: From the CTC decoder, representing certainty about the recognized characters in line $i$.

**Document-level average confidence:**

$$
\bar{c} = \frac{1}{n} \sum_{i=1}^{n} c_i
$$

**Quality thresholds used in the pipeline:**

| Threshold | Interpretation | Pipeline Action |
|-----------|---------------|-----------------|
| $\bar{c} \geq 0.85$ | High quality | Proceed normally |
| $0.65 \leq \bar{c} < 0.85$ | Acceptable | Proceed, log info |
| $\bar{c} < 0.65$ | Low quality | Proceed with warning, flag for human review |
| $\bar{c} = 0$ (empty) | No text detected | Classify as UNKNOWN (via threshold, not if/else) |

**Common failure modes and mitigations:**

| Symptom | Root Cause | Mitigation |
|---------|-----------|------------|
| $\bar{c} < 0.5$ with garbled text | Heavily degraded scan, low resolution | Increase scan DPI to 300+, apply denoising |
| Empty result, $n = 0$ lines | Image contains no text (photo, diagram) | Pipeline gracefully returns UNKNOWN |
| Systematic misrecognition | Wrong language model loaded | Verify `ocr_lang` setting |
| Missing text regions | `det_db_box_thresh` too high | Lower to 0.3 for faint text |
| Upside-down text recognized | `use_angle_cls=False` | Always enable angle classification |

### 4.6 PDF Extraction Strategy: Native vs OCR

The pipeline implements a **dual-strategy** for PDFs:

```python
for page in pdf.pages:
    native_text = page.get_text()     # PyMuPDF native extraction
    if len(native_text.strip()) > 50:
        use native_text               # Digital PDF → no OCR needed
    else:
        rasterize at 300 DPI          # Scanned page → OCR
        use PaddleOCR result
```

**Why 50 characters as the threshold?** A page with fewer than ~50 characters of extractable text is almost certainly a scanned image or a page with embedded images. Headers/footers alone rarely exceed this threshold. This is a pragmatic heuristic applied to the extraction method selection — not to the document content classification (which remains zero-fragility).

**DPI selection:** We rasterize at 300 DPI (configurable), which provides a good balance:

| DPI | Resolution (A4) | OCR Quality | Speed |
|-----|-----------------|-------------|-------|
| 150 | 1240 × 1754 | Marginal | ~150ms/page |
| 200 | 1654 × 2339 | Good | ~250ms/page |
| **300** | **2480 × 3508** | **Excellent** | **~400ms/page** |
| 600 | 4960 × 7016 | Diminishing returns | ~1200ms/page |

---

## 5. Embedding Engine: E5-Large Multilingual

### 5.1 Architecture & Training Objective

`intfloat/multilingual-e5-large` is a **560M-parameter** transformer model based on XLM-RoBERTa-Large, fine-tuned specifically for text embedding tasks.

**Key specs:**

| Property | Value |
|----------|-------|
| Parameters | 560M |
| Base model | XLM-RoBERTa-Large |
| Embedding dimension | 1024 |
| Max sequence length | 514 tokens |
| Languages supported | 100+ (including French, English, Wolof) |
| MTEB average score | 64.4 |

**Training pipeline:** E5 models undergo three-stage training:

1. **Weakly supervised pre-training** on ~1B text pairs from web data (e.g., title-body, question-answer pairs scraped at scale)
2. **Supervised fine-tuning** on high-quality labeled datasets: MS MARCO, Natural Questions, NLI collections, multilingual parallel corpora
3. **Instruction tuning** to differentiate query vs passage encoding behavior via prefix conditioning

### 5.2 Contrastive Learning: InfoNCE Loss

The core training objective is a contrastive loss that teaches the model to place semantically similar texts close together and dissimilar texts far apart.

Given a query $q$, its positive document $d^+$, and a set of $N-1$ in-batch negatives $\{d_1^-, d_2^-, \ldots, d_{N-1}^-\}$, the **InfoNCE loss** is:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\big(\text{sim}(q, d^+) / \tau\big)}{\exp\big(\text{sim}(q, d^+) / \tau\big) + \displaystyle\sum_{j=1}^{N-1} \exp\big(\text{sim}(q, d_j^-) / \tau\big)}
$$

where:
- $\text{sim}(a, b) = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}$ (cosine similarity)
- $\tau$ is the temperature parameter (typically $\tau \approx 0.01 - 0.05$), which controls the sharpness of the probability distribution

**Intuition:** This loss is a $N$-way softmax classifier that tries to identify the positive document among $N$ candidates. Lower temperature $\tau$ makes the distribution peakier, forcing the model to learn finer-grained distinctions.

**In-batch negatives:** With a batch size of $B$, each query uses the other $B-1$ positive documents as negatives. This provides $O(B^2)$ training signal per batch — a key reason why large batch sizes improve embedding quality.

### 5.3 The Critical Role of Prefixes

E5 models require specific prefixes to produce optimal embeddings:

| Use Case | Prefix | Encoding Strategy |
|----------|--------|-------------------|
| Indexing documents / passages | `passage: ` | Emphasizes information content, longer text |
| Searching / querying / classifying | `query: ` | Emphasizes intent, shorter text |

**Why different prefixes?** During training, queries and passages have fundamentally different distributions:

- **Queries** are short, intent-focused ("what is the maintenance cost?")
- **Passages** are longer, information-rich ("The maintenance contract specifies a monthly cost of...")

The prefix acts as a **conditional signal** that tells the model which encoding strategy to use. The transformer's self-attention mechanism adapts its behavior based on this prefix. Ablation studies show that omitting prefixes degrades retrieval metrics (nDCG@10) by 5-15%.

**In our pipeline:**
- **Route descriptions** → encoded with `passage:` (they define reference documents for each class)
- **Document text to classify** → encoded with `query:` (we are querying "which class?")

This asymmetry is intentional and handled automatically by the `EmbeddingEngine` class.

### 5.4 L2 Normalization & Unit Hypersphere

After the transformer produces a raw embedding $\mathbf{v} \in \mathbb{R}^{1024}$, we normalize it:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \quad \Longrightarrow \quad \hat{\mathbf{v}} \in \mathcal{S}^{1023}
$$

where $\mathcal{S}^{1023}$ is the unit hypersphere in 1024 dimensions. All embeddings live on the surface of this sphere.

**Geometric picture:** The "distance" between two points on the unit hypersphere is entirely determined by the angle between them:

$$
d_{\text{geodesic}}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \arccos(\hat{\mathbf{a}} \cdot \hat{\mathbf{b}}) = \theta
$$

The full complexity of semantic similarity is thus captured by a single scalar: the angle $\theta \in [0, \pi]$ between embedding vectors.

### 5.5 Average Pooling: From Tokens to Sentences

The transformer processes a sequence of $L$ tokens and produces a hidden state $\mathbf{H} \in \mathbb{R}^{L \times d}$ where $d = 1024$. To obtain a single embedding vector for the entire text, we apply **masked average pooling**:

$$
\mathbf{v} = \frac{\sum_{i=1}^{L} m_i \cdot \mathbf{h}_i}{\sum_{i=1}^{L} m_i}
$$

where $m_i \in \{0, 1\}$ is the attention mask (1 for real tokens, 0 for padding).

**Why average pooling over [CLS] token?** Empirically, average pooling consistently outperforms [CLS]-based pooling for sentence embedding tasks because:

1. It aggregates information from all token positions, not just the first
2. It is more robust to positional biases in the transformer
3. It produces embeddings with lower variance across runs

The `SentenceTransformer` library handles this pooling internally, but understanding it is important for debugging and custom implementations.

### 5.6 Batch Processing & GPU Optimization

The embedding engine processes texts in batches to maximize GPU throughput:

```
N texts → Tokenizer (pad to max length) → Tensor (N, L) → GPU forward → N × (L, 1024) → Pool → N × 1024
```

**Key parameters and their effect:**

| Parameter | Default | Effect on Throughput | Effect on VRAM |
|-----------|---------|---------------------|----------------|
| `batch_size` | 32 | ↑ batch = ↑ throughput (until VRAM limit) | Linear increase |
| `max_seq_length` | 512 | ↑ length = ↓ throughput (quadratic attention) | Quadratic increase |

**VRAM estimation formula:**

$$
\text{VRAM} \approx \text{model\_size} + B \times L \times d \times 4 \times 3
$$

where $B$ = batch size, $L$ = sequence length, $d$ = hidden dim, $\times 4$ for float32 bytes, $\times 3$ for activations + gradients + optimizer states (inference only needs $\times 1$).

For E5-large with $B=32$, $L=512$: approximately 4.5 GB for the model + 0.2 GB for activations = ~4.7 GB total during inference.

**Throughput benchmarks:**

| GPU | batch=16 | batch=32 | batch=64 |
|-----|----------|----------|----------|
| T4 (16GB) | ~150 docs/s | ~200 docs/s | OOM |
| A10G (24GB) | ~250 docs/s | ~350 docs/s | ~400 docs/s |
| A100 (40GB) | ~350 docs/s | ~500 docs/s | ~650 docs/s |

---

## 6. Semantic Router: How Classification Works

The `SemanticRouter` combines all the theory above into a single, elegant classifier:

```
                  INIT (once at startup)                 INFERENCE (per document)
    ┌─────────────────────────────────┐    ┌──────────────────────────────────────┐
    │                                 │    │                                      │
    │  Route descriptions (N=4-8     │    │  Document text                       │
    │  per class, K=7 classes)       │    │  ┌───────────────────────┐           │
    │  ┌────────────────────────┐    │    │  │ "FACTURE N°2024       │           │
    │  │ "Invoice with line     │    │    │  │  Client: SARL Tech    │           │
    │  │  items, total amount   │    │    │  │  Total TTC: 15300€    │           │
    │  │  due, payment terms"   │    │    │  │  Échéance: 30 jours"  │           │
    │  └───────────┬────────────┘    │    │  └──────────┬────────────┘           │
    │              │                 │    │             │                         │
    │        embed_documents()       │    │       embed_query()                   │
    │        prefix: "passage:"      │    │       prefix: "query:"               │
    │              │                 │    │             │                         │
    │        ┌─────▼─────┐           │    │       ┌─────▼─────┐                  │
    │        │ embeddings│           │    │       │ query_vec │                  │
    │        │ (N, 1024) │           │    │       │  (1024,)  │                  │
    │        └─────┬─────┘           │    │       └─────┬─────┘                  │
    │              │                 │    │             │                         │
    │        mean + L2 norm          │    │    ┌────────▼─────────┐               │
    │              │                 │    │    │                  │               │
    │        ┌─────▼─────┐           │    │    │  q @ C.T         │               │
    │        │ centroid_k │──────────│────│──> │  = similarities  │               │
    │        │  (1024,)   │          │    │    │    (K,)          │               │
    │        └───────────┘           │    │    └────────┬─────────┘               │
    │                                │    │             │                         │
    │  centroids C: (K, 1024)        │    │       argmax + τ                     │
    │  stored in memory              │    │             │                         │
    │                                │    │    ┌────────▼─────────┐               │
    └─────────────────────────────────┘    │    │ FACTURE          │               │
                                           │    │ confidence=0.847 │               │
                                           │    │ margin=0.213     │               │
                                           │    └──────────────────┘               │
                                           └──────────────────────────────────────┘
```

**Time complexity summary:**

| Phase | Complexity | Typical Time |
|-------|-----------|-------------|
| Init: embed $M$ descriptions | $O(M \cdot L^2 \cdot d)$ | ~2s (one-time) |
| Init: compute $K$ centroids | $O(K \cdot m \cdot d)$ | <1ms |
| Classify 1 document | $O(L^2 \cdot d + K \cdot d)$ | ~25ms |
| Classify $n$ documents (batch) | $O(n \cdot L^2 \cdot d + n \cdot K \cdot d)$ | ~200ms for $n=100$ |

---

## 7. Project Structure

```
doc_classifier/
├── config/
│   ├── __init__.py
│   ├── settings.py              # Centralized Pydantic Settings
│   └── routes.py                # Semantic route definitions
├── engines/
│   ├── __init__.py
│   ├── ocr_engine.py            # PaddleOCR wrapper (hardened)
│   └── embedding_engine.py      # SentenceTransformer E5 wrapper
├── classifier/
│   ├── __init__.py
│   ├── models.py                # Pydantic I/O models (strict validation)
│   ├── semantic_router.py       # Core: cosine similarity classification
│   └── pipeline.py              # Orchestrator: OCR → Embed → Classify
├── tests/
│   ├── __init__.py
│   └── test_router.py           # Unit tests for SemanticRouter
├── main.py                      # FastAPI endpoints
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

**Separation of concerns:**

| Module | Responsibility | Depends On |
|--------|---------------|------------|
| `config/` | Settings validation, route definitions | Nothing |
| `engines/` | Raw I/O with ML models (OCR, embeddings) | `config/` |
| `classifier/` | Classification logic and pipeline orchestration | `config/`, `engines/` |
| `main.py` | HTTP API layer (thin wrapper) | `classifier/` |

---

## 8. Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Classify a document
curl -X POST http://localhost:8000/classify \
  -F "file=@invoice_scan.pdf"
```

### Docker

```bash
# Build and run (requires NVIDIA Docker runtime for GPU)
docker compose up -d

# Health check
curl http://localhost:8000/health

# Classify a single document
curl -X POST http://localhost:8000/classify \
  -F "file=@contract.pdf"

# Batch classify
curl -X POST http://localhost:8000/classify/batch \
  -F "files=@doc1.pdf" -F "files=@doc2.pdf" -F "files=@doc3.pdf"
```

### Python API (Direct Usage)

```python
from pathlib import Path
from classifier.pipeline import ClassificationPipeline

# Initialize once (loads OCR + embedding models into GPU)
pipeline = ClassificationPipeline()

# Single file classification
result = pipeline.classify_file(Path("scan_contrat.pdf"))
print(f"Type:       {result.classification.document_type.value}")
print(f"Confidence: {result.classification.confidence:.3f}")
print(f"Margin:     {result.classification.margin:.3f}")
print(f"OCR conf:   {result.ocr_confidence:.2f}")
print(f"Latency:    {result.total_latency_ms:.0f}ms")

# Batch processing (GPU-optimized)
batch = pipeline.classify_batch([
    Path("facture_001.pdf"),
    Path("contrat_002.pdf"),
    Path("rapport_003.docx"),
    Path("scan_004.png"),
])
print(f"Throughput: {batch.throughput_docs_per_sec:.0f} docs/sec")
for r in batch.results:
    print(f"  {r.file_path}: {r.classification.document_type.value} "
          f"({r.classification.confidence:.3f})")
```

### Debugging & Explainability

```python
from classifier.semantic_router import SemanticRouter

router = SemanticRouter()

# Get detailed classification breakdown
explanation = router.explain(
    "FACTURE N° 2024-0847 — Client: SARL Technologie — Total TTC: 15 300,00€"
)
print(explanation)
# {
#     "predicted": "facture",
#     "confidence": 0.8472,
#     "threshold": 0.40,
#     "is_confident": True,
#     "margin": 0.2134,
#     "all_scores": {
#         "facture": 0.8472,
#         "contrat": 0.6338,
#         "rapport": 0.5921,
#         "formulaire": 0.5103,
#         "courrier": 0.4887,
#         "ressources_humaines": 0.4562,
#         "technique": 0.4201
#     },
#     "model": "intfloat/multilingual-e5-large"
# }
```

---

## 9. API Reference

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/classify` | POST | Single file (multipart upload) | `PipelineResult` |
| `/classify/batch` | POST | Multiple files (multipart upload) | `BatchResult` |
| `/classify/explain` | POST | JSON `{"text": "..."}` | Detailed score breakdown |
| `/health` | GET | — | Model info, device, routes |
| `/routes` | GET | — | All configured semantic routes |

---

## 10. Configuration

All settings via environment variables (prefix `DIP_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DIP_EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-large` | HuggingFace model |
| `DIP_EMBEDDING_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `DIP_EMBEDDING_BATCH_SIZE` | `32` | GPU batch size |
| `DIP_OCR_LANG` | `fr` | PaddleOCR language |
| `DIP_OCR_USE_GPU` | `true` | GPU for OCR |
| `DIP_OCR_USE_ANGLE_CLS` | `true` | Rotation detection |
| `DIP_CONFIDENCE_THRESHOLD` | `0.40` | Rejection threshold |
| `DIP_CLASSIFICATION_TOP_K` | `3` | Candidate classes returned |

---

## 11. Adding New Document Classes

Edit **only** `config/routes.py`:

```python
# Step 1: Add to the enum
class DocumentType(str, Enum):
    # ... existing types ...
    DEVIS = "devis"  # ← new

# Step 2: Add a RoutePrototype
ROUTES.append(
    RoutePrototype(
        document_type=DocumentType.DEVIS,
        descriptions=(
            "Proposition commerciale chiffrée avec détail des prestations et prix",
            "Cost estimate or quotation with line items and validity period",
            "Devis détaillé avec conditions de vente et délai de validité",
            "Commercial offer listing services, unit prices and total cost",
        ),
    )
)
```

**No code changes anywhere else.** The centroid is computed automatically at startup.

---

## 12. Testing

```bash
pytest tests/ -v
pytest tests/ --cov=classifier --cov-report=term-missing
```

---

## 13. Deployment

```bash
docker compose up -d
```

The Dockerfile pre-downloads E5-large into the image so the first request is fast.

---

## 14. Performance Benchmarks

| Metric | T4 (16GB) | A10G (24GB) | A100 (40GB) |
|--------|-----------|-------------|-------------|
| Model load | ~8s | ~5s | ~3s |
| Single classify | ~50ms | ~35ms | ~25ms |
| Batch (100 docs) | ~500ms | ~300ms | ~200ms |
| Throughput | ~200/s | ~350/s | ~500/s |
| VRAM | ~4.5GB | ~4.5GB | ~4.5GB |

OCR adds 100-500ms per page depending on complexity and DPI.

---

*Internal use — Orange DATA-IA / SND / DREAMS*

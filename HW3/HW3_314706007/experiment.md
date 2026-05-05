# HW3 Progress Summary (Updated)

## Experiment Timeline

| # | What changed | Dev F1 | Kaggle |
|---|-------------|--------|--------|
| 0 | Initial broken pipeline (logit-based, broken masking) | 0.115 | — |
| 1 | Fix label masking, switch to text generation, CoT prompt | — | 0.67 |
| 2 | + focal loss γ=2 + few-shot examples | 0.55 | — |
| 3 | Revert focal+few-shot, switch BM25 → pure embedding | 0.61 | — |
| 4 | Hybrid BM25+embedding (RRF), max_seq_len shrink, evidence 350w | 0.45 | — |
| 5 | Layer 1+2: 550w evidence + max_seq_len 3072 + trimmed prompt + ref stripping + sentence-window chunks + multi-query retrieval (BM25 keys) + decision rules + AF weight ×0.7 + sliding chunks + review cap | 0.7474 | 0.74 |
| 6 | Layer A (entity rule tightened, Overgen universals, AF ×0.85, keys 0.5) | 0.7353 | — |
| 7 | Layer A revert + Layer C augmentation (paraphrased Number/Temporal) | 0.7456 | — |
| 8 | Step 1+3: keys 0.5→0.3, AF ×0.92, cross-encoder rerank | **0.7598** | — |
| 9 | + Method A (hybrid pack: 3 reranked + 2 RRF) + Method D (self-verify) | 0.7565 | — |
| 10 | **Method B (contrastive CoT training)** | in progress | — |

---

## Methods Summary

### ✅ What helped (kept)

| Method | Where | Effect |
|--------|------|--------|
| Prompt-length-based label masking | utils + train | Foundation; without this nothing trains |
| Text-generation inference (assignment-compliant) | inference | Required compliance + accuracy |
| Decision rules in system prompt ("X → NOT AF") | utils prompt | Single biggest gain; broke AF default-class bias |
| Multi-query hybrid retrieval (BM25 + embedding + BM25-keys, RRF) | utils retriever | Lexical + semantic anchors |
| Sentence-window chunks + noise filter | utils PDF parser | Finer retrievable units |
| References stripping in PDF | utils PDF parser | Removed entity-shaped BM25 noise |
| Review text capping (400 words) | utils PromptBuilder | Prevented prompt overflow |
| AF class weight × 0.92 | train | Mild rebalance of AF over-prediction |
| LoRA r=32, α=64 | train | Capacity for 5-way classification |
| max_seq_len=3072, evidence 550w | train+inference | Sweet spot for time vs context |
| Cross-encoder reranking | utils retriever | Killed Number under-prediction; +0.014 F1 |
| Hybrid evidence packing (3 rerank + 2 RRF) | utils retriever | Limited effect; insufficient Overgen improvement |

---

### ❌ What hurt (rejected)

| Method | Why it failed |
|--------|--------------|
| Focal loss γ=2 on token-level CoT | Per-token gradient suppression killed effective learning (−13 pp) |
| Few-shot examples in training prompt | Memorized as canned templates; over-narrowed the concept |
| Pure embedding retrieval (no BM25) | Lost lexical Entity-matching anchor |
| Evidence budget 350w | Entity recall collapsed; entity-bearing paragraph not retrieved |
| Universals list ("all/always/every") in Overgen rule | Narrowed Overgen concept; lost correct predictions |
| Multi-step Entity exception in prompt | Model ignored the conditional |
| AF weight × 0.7 (too aggressive) | Over-corrected AF from +14pp to −4pp |

---

### ⚖️ Neutral / inconclusive

| Method | Result |
|--------|--------|
| Layer C (Number/Temporal paraphrasing) | Marginal effect (minimal class change) |
| Method D (self-verification, pre-contrastive) | Near-zero gain; template outputs limit usefulness |

---

## Current Best Confirmed

- **Best Dev F1:** 0.7598 (configuration #8)  
- **Best Kaggle F1:** 0.74 (configuration #5)  
- **Method A + D (#9):** 0.7565 (−0.003 regression)

---

## Key Diagnosis

The system is limited by a **template-based reasoning ceiling**:

- Model emits **one fixed response per class**
- No true reasoning → only pattern matching
- Inference-side improvements (Method A, D) show diminishing or negative returns

Evidence:
- Verification flipped only **17 / 1777 (0.96%)**
- Overgen decreased (486 → 481)
- Net F1 dropped despite added methods

---

## New Method — Method B (Contrastive CoT Training)

### What changed

| Area | Before | After |
|------|--------|--------|
| Training CoT | 5 fixed templates per class | Sample-specific contrastive reasoning |
| Token diversity | Identical outputs per class | Dynamic tokens (entity/number extraction) |
| Reasoning | Single-class explanation | Explicit "NOT X" discrimination |
| max_new_tokens | 150 | 220 |

---

### Core idea

Force the model to **explicitly distinguish the correct class from all incorrect ones**.

Example (Entity):

Old:
> Template-based, identical across samples

New:
> Mentions extracted entities and explicitly rejects other classes

---

### Expected impact

- Removes template memorization shortcut  
- Improves class boundary learning  
- Enables downstream reasoning-based methods  

---

## Interaction with Method D

Before:
- Verifier ineffective due to lack of reasoning signal  

After Method B:
- Outputs contain explicit discrimination logic  
- Verifier can perform meaningful re-evaluation  

Expected:
- Flip rate increases (**17 → 50–150**)  
- Higher proportion of correct corrections  

---

## Remaining Methods

### Tier 1 — highest expected gain

- **Method B (contrastive CoT training)**  
  Expected: +0.02–0.03  

- **Method C (two-stage cascade classifier)**  
  Expected: +0.03–0.05  

---

### Tier 2 — inference-side

- **Method E (class-prior-aware parser)**  
  Expected: +0.005–0.015  

---

## Realistic Projection to 0.82

| Path | Expected Dev F1 |
|------|----------------|
| Current (#9) | 0.756–0.76 |
| + Method B | 0.78–0.80 |
| + Method C | 0.81–0.83 |
| + Method E | tie-breaker |

---

## Final Strategy

The clearest path forward is:

> **Method B → Method C → Method E**

- Method B fixes the **core training limitation**  
- Method C improves **decision structure**  
- Method E provides **low-cost final gains**  

---

## Key Takeaway

The bottleneck is not retrieval or inference, but **training signal quality**.

> Transitioning from **template-based CoT → contrastive CoT** is the critical step to unlock further performance gains.
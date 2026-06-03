# Research Overview: Bridging the Language Gap in Cross-lingual RAG for Indonesian–English Question Answering

> **Target:** F1 ≥ 0.5 → 0.80 (progressive milestone; baseline peaks at ~0.41 F1 on easy datasets and collapses to < 0.01 on multi-hop)

---

## 0. Research Phase Overview

| #   | Phase                                 | Dataset                                                             | Proses Singkat                                                                                                                                                                                                  | Arsitektur                                                            | Metrik                                                                    |
| --- | ------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 0   | **Data Preparation**                  | HotpotQA, XQuAD-ID, TyDiQA-ID                                       | Download, filter missing pairs; HotpotQA: ~10.000 train + 7.405 eval (label "C" hardcoded); TyDiQA-ID: split stratified 80/20 → 4.815 train + 1.452 eval; XQuAD-ID: 1.190 full untuk eval saja (tidak di-split) | —                                                                     | —                                                                         |
| 1   | **Complexity Classifier Fine-tuning** | Train: HotpotQA (~10.000, class C) + TyDiQA-ID (4.815, class A/B/C) | Fine-tune mDeBERTa-v3-base dengan weighted loss (kompensasi class imbalance karena HotpotQA dominan di class C); XQuAD-ID tidak dipakai untuk training                                                          | mDeBERTa-v3-base                                                      | Accuracy, Precision, Recall, F1, Confusion Matrix                         |
| 2   | **Baseline Reproduction**             | HotpotQA eval only                                                  | Replikasi hasil baseline pada HotpotQA — kasus kegagalan paling ekstrem (multi-retrieval collapse 0.007); tidak memerlukan classifier karena semua pertanyaan langsung dilabeli "C"                             | BM25 Index; no classifier for HotpotQA                                | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                    |
| 3   | **Ablation Study**                    | HotpotQA eval, XQuAD-ID eval (ID query & EN query dipisah)          | Evaluasi progresif per config; tiap config dijalankan pada dataset-native corpus                                                                                                                                | BM25 → BGE-M3 Dense → BGE-M3 Sparse+Dense → +mMiniLM Reranker → +CRAG | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                    |
| 4   | **Δgap Measurement**                  | XQuAD-ID eval, TyDiQA-ID eval; corpus: Wikipedia ID                 | Query ID dan EN identik dijalankan pada Wikipedia ID corpus; hitung selisih per retrieval config                                                                                                                | BGE-M3 Sparse+Dense + mMiniLM Reranker                                | Recall@10, MRR, Δgap per config (NDCG@10 dropped — binary relevance only) |
| 5   | **RAGAS Evaluation**                  | XQuAD-ID eval, TyDiQA-ID eval; corpus: Wikipedia ID                 | Evaluasi grounding kualitas jawaban full-system                                                                                                                                                                 | Full System (BGE-M3 + Reranker + CRAG + Reasoning LLM)                | Faithfulness, Factual Correctness, Semantic Similarity                    |

**Catatan corpus per phase:**

- Phase 2 (Baseline): HotpotQA source docs saja, diindeks ke BM25 Index. XQuAD-ID tidak dipakai di fase ini.
- Phase 3 (Ablation): dokumen bawaan dataset masing-masing (HotpotQA source docs EN; XQuAD-ID passage bawaan). EM/F1 dihitung karena ground truth tersedia.
- Phase 4 & 5 (Δgap + RAGAS): Wikipedia ID (~20.000 artikel) — open-domain, cross-lingual. Alasan: dataset-native passages artificially inflate retrieval scores karena sudah semantically aligned ke pertanyaannya. Wikipedia ID mengeliminasi confound ini dan merepresentasikan real-world open-domain retrieval scenario yang menjadi target arsitektur ini.

---

## 1. Literature Review & Paper Analysis

### Paper 1 — Baseline (Primary Target for Improvement)

**"Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering"**
_William Christian et al., Bina Nusantara University — arXiv:2510.21068v1, Oct 2025_

**What they did:**

- Translated HotpotQA (EN→ID) using OPUS-MT without manual post-editing — introduces semantic drift and grammatical errors
- Retrieval via BM25 (ElasticSearch) — sparse, keyword-only, no cross-lingual capability
- Question complexity classifier: fine-tuned IndoBERT (base/large, seq 128/512); best model = IndoBERT Large P1 at 0.756 accuracy — **valid only on IndoQA/QASiNa distribution** (see note below)
- Three answering strategies: non-retrieval (A), single-retrieval (B), multi-retrieval via IRCoT (C)
- LLMs tested: Gemma 3-4B, Qwen 3-8B (local, constrained GPU)

**Measured results:**

| Dataset  | Best Method      | Model   | Accuracy | F1    |
| -------- | ---------------- | ------- | -------- | ----- |
| IndoQA   | Single Retrieval | Qwen 3  | 0.264    | 0.265 |
| QASiNa   | Single Retrieval | Qwen 3  | 0.411    | 0.411 |
| HotpotQA | Non-Retrieval    | Gemma 3 | 0.058    | 0.059 |

**Critical failures identified:**

- Multi-retrieval collapses on IndoQA: Gemma 3 drops from 0.258 (single) → 0.018 (multi), **≈14× degradation**. On HotpotQA: 0.055 (single) → 0.007 (multi), ≈**7.8× degradation**. Keduanya harus dikutip dengan dataset yang benar — 14× hanya berlaku untuk IndoQA.
- IRCoT multi-retrieval menggunakan termination condition yang lemah: berhenti jika LLM menghasilkan kata "Jawaban" _atau_ setelah maksimum 5 iterasi. Tidak ada mekanisme evaluasi kualitas retrieval per iterasi atau confidence gating. Loop tidak berjalan tanpa batas (_bukan "blind loop"_), tapi terminasinya tidak principled — mudah di-trigger prematur atau gagal berhenti saat halusinasi sudah terjadi.
- Penyebab kegagalan yang diakui paper: LLM kecil (4–8B) dengan kemampuan reasoning Bahasa Indonesia terbatas; context window membengkak karena akumulasi dokumen tanpa filtering; kombinasi keduanya memperparah halusinasi. Ini bukan semata-mata masalah desain IRCoT.
- OPUS-MT translation noise: "three-time Tony nominee" → "tiga kali Tony nominasi" (semantic drift, bukan kegagalan total)
- Tidak ada reranker — BM25 retrieval score tidak dikalibrasi lintas bahasa
- Tidak ada pengukuran language gap sama sekali

**Catatan metodologis pada classifier baseline:**

HotpotQA di-bypass dari proses labeling dan seluruh pertanyaannya langsung di-assign label "C" tanpa menjalankan pipeline annotation. Classifier IndoBERT di-fine-tune dari _labeled data obtained from the annotation process_ — artinya kemungkinan besar hanya menggunakan IndoQA + QASiNa. Namun paper tidak mengungkapkan secara eksplisit apakah training data HotpotQA ("C" hardcoded) diikutsertakan dalam fine-tuning.

Implikasinya: klaim **0.756 accuracy classifier tidak dapat digeneralisasi ke HotpotQA**. Angka tersebut valid hanya untuk distribusi IndoQA/QASiNa. Saat adaptive retrieval dijalankan pada HotpotQA, paper menyebutkan "almost always selects multi-retrieval" — kata "almost always" (bukan "always") mengindikasikan classifier tetap dijalankan, bukan di-hardcode saat evaluasi, namun generalitasnya pada pertanyaan multi-hop HotpotQA tidak pernah divalidasi.

**What is explicitly admitted as future work (our direct attack surface):**

- Native Indonesian multi-hop datasets
- Indonesian-specialized or multilingual LLM for reasoning
- Improved multi-retrieval with a principled stopping criterion

---

### Paper 2 — Strong Supporting Evidence

**"Performance Analysis of Reasoning Models in RAG-Based QA for University Admission Services"**
_Setiawan et al., UPN Veteran Jatim — Bit-Tech Vol.8 No.3, April 2026_

**What they did:**

- Built a multilingual RAG system (Indonesian, English, Javanese) over 30 official institutional documents
- Dense vector retrieval: VoyageAI embeddings → ChromaDB (cosine similarity)
- Query rewriting with 3 variations to expand semantic search space
- Compared reasoning vs. non-reasoning models under identical retrieval conditions
- Evaluation via RAGAS framework: faithfulness, factual correctness, semantic similarity

**Measured results:**

| Model            | Type          | Faithfulness | Factual Correctness | Semantic Similarity | Avg        |
| ---------------- | ------------- | ------------ | ------------------- | ------------------- | ---------- |
| Gemini-2.5-Flash | Reasoning     | 0.9195       | 0.6274              | 0.9151              | **0.8207** |
| DeepSeek-R1      | Reasoning     | 0.8302       | 0.5457              | 0.9067              | 0.7609     |
| o4-mini          | Reasoning     | 0.8251       | 0.5299              | 0.8948              | 0.7499     |
| DeepSeek-V3      | Non-Reasoning | 0.8367       | 0.5309              | 0.8983              | 0.7553     |
| GPT-4o-mini      | Non-Reasoning | 0.7783       | 0.4537              | 0.8870              | 0.7063     |

**Key findings applicable to our work:**

- Reasoning models outperform non-reasoning by +6.63% overall and +15.95% on factual correctness
- Dense retrieval (vector-based) achieves stable multilingual performance
- English queries still outperform Indonesian — confirms a real language gap that needs to be measured
- Retrieval quality is the dominant success factor

---

### Paper 3 — Foundational Reference

**"Utilizing RAG in LLMs to Enhance Indonesian Language NLP"**
_Tohir, Merlina, Haris — Universitas Nusa Mandiri, JITK Vol.10 No.2, Nov 2024_

**What they did:**

- Applied a standard RAG pipeline (chunk → vector store → retrieve → generate) on an Indonesian-language religious corpus (Quran 30 Juz)
- GPT-4-based generation with Indonesian retrieval context

**Key takeaway:** RAG significantly improves Indonesian NLP tasks compared to LLM-only baselines. Demonstrates the foundational viability of the approach for Indonesian-language domains.

---

## 2. Gap Analysis Summary

| Gap                             | Baseline Paper (2510.21068)                                                                            | Evidence from Supporting Papers                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Retrieval method**            | BM25 sparse only                                                                                       | Paper 2: dense (VoyageAI) enables stable multilingual retrieval                                  |
| **Translation noise**           | OPUS-MT introduces semantic drift                                                                      | Resolved by eliminating translation — BGE-M3 encodes cross-lingually without translation         |
| **Multi-retrieval failure**     | IRCoT terminates by keyword/max-5 without confidence gating; small LLMs hallucinate under long context | No fix proposed; explicitly listed as future work                                                |
| **No reranker**                 | Skipped entirely                                                                                       | Critical in cross-lingual settings where embedding similarity is not calibrated across languages |
| **Language gap unmeasured**     | No Δgap metric                                                                                         | Paper 2 shows EN > ID gap exists; not quantified                                                 |
| **BM25 language asymmetry**     | BM25 used for all query types                                                                          | BM25 fails on EN→ID retrieval; BGE-M3 sparse mode is the cross-lingual alternative               |
| **LLM reasoning**               | Small models (4B–8B), non-reasoning                                                                    | Paper 2: reasoning models give +15.95% factual correctness                                       |
| **Classifier generalizability** | Classifier trained on IndoQA/QASiNa; validity on HotpotQA unvalidated                                  | Not addressed in any reviewed paper                                                              |
| **Evaluation framework**        | EM + F1 only                                                                                           | Paper 2: RAGAS gives richer grounding-aware assessment                                           |

---

## 3. Proposed Improved Pipeline

### 3.1 Architecture Overview

```
Query Input (Indonesian OR English)
        │
        ▼
┌──────────────────────────────────────────┐
│  Bilingual Query Processor               │
│  - Language detector (ID / EN)           │
│  - mDeBERTa-v3-base complexity           │
│    classifier (A / B / C)                │
│  NOTE: NO translation step — BGE-M3      │
│  handles cross-lingual natively          │
└─────────┬────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                    Retrieval Layer                           │
│                                                              │
│  [BM25 + Sastrawi]   [BGE-M3 Dense]   [BGE-M3 Sparse+Dense] │
│  ID-only, ablation   Multilingual      Recommended config    │
│  baseline            100+ languages    (1 model, 2 modes)    │
│                                        via RRF fusion        │
│                                                              │
│  ⚠ Hybrid BM25+BGE-M3: ablation only — BM25 injects noise   │
│    on EN queries due to language mismatch                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  mMiniLM Reranker     │  cross-encoder/mmarco-mMiniLMv2-L12-H384
               │  (language-agnostic   │  scores (query, doc) pairs
               │   relevance scoring)  │  applied after top-k retrieval
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Adaptive Generation  │
               │  A → No retrieval     │
               │  B → Single-shot      │
               │  C → CRAG             │  Confidence-gated by LLM itself;
               │                       │  context managed via replace +
               └───────────┬───────────┘  carry-over 1 iterasi
                           │
                           ▼
                  Answer (Indonesian)
```

### 3.2 Component Specifications

#### Input & Query Processing

- Accept native Indonesian or English queries without any translation step
- Language detector (e.g., `langdetect` atau `lingua-py`) untuk tag query sebagai ID atau EN
- mDeBERTa-v3-base complexity classifier — fine-tuned pada train split ketiga dataset (lihat Section 3.3)
- **Tidak ada translation engine** — BGE-M3 mengenkode kedua bahasa ke dalam shared multilingual embedding space, mengeliminasi bottleneck OPUS-MT pada baseline

#### Dataset

| Dataset   | Split                      | N (train)             | N (eval)     | Phase 2 & 3                       | Phase 4 (Δgap) | Phase 5 (RAGAS) | Keterangan                                                                                                               |
| --------- | -------------------------- | --------------------- | ------------ | --------------------------------- | -------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| HotpotQA  | train ~10.000 / eval 7.405 | ~10.000 (all class C) | 7.405        | ✓ eval (Phase 2 & 3)              | —              | —               | Multi-hop EN; label C hardcoded, weighted loss untuk kompensasi imbalance                                                |
| TyDiQA-ID | stratified 80/20           | 4.815                 | 1.452        | ✗ tidak dipakai                   | ✓ eval         | ✓ eval          | Native ID; dikeluarkan dari ablation untuk menghindari distributional familiarity                                        |
| XQuAD-ID  | **tidak di-split**         | —                     | 1.190 (full) | ✓ eval (Phase 3, EN & ID dipisah) | ✓ eval         | ✓ eval          | Parallel EN/ID pairs; full dipakai untuk eval saja — split akan mengorbankan parallel structure yang kritikal untuk Δgap |

**Mengapa XQuAD-ID tidak di-split untuk training:**
XQuAD-ID terdiri dari 1.190 parallel EN/ID question pairs — tiap pertanyaan tersedia dalam dua bahasa identik secara semantik. Struktur ini adalah satu-satunya sumber untuk mengukur Δgap secara bersih (query yang identik, corpus yang sama, retrieval config yang sama). Memotong sebagian ke training classifier akan mengurangi coverage evaluasi Δgap dan memperlemah klaim cross-lingual kontribusi ini. Oleh karena itu, XQuAD-ID dipakai full 1.190 untuk evaluasi saja.

**Mengapa TyDiQA-ID dikeluarkan dari ablation (Phase 2 & 3):**
mDeBERTa-v3-base classifier di-fine-tune menggunakan 4.815 train instance TyDiQA-ID. Meskipun tidak ada instance overlap dengan eval split (1.452), classifier sudah familiar dengan pola distribusi pertanyaan dari dataset yang sama. Memakai eval split TyDiQA-ID di ablation membuka risiko distributional familiarity yang sulit didefend ke reviewer.

**Acknowledged limitation — class C coverage pada ID queries:**
Training data classifier didominasi oleh HotpotQA (class C, English) dan TyDiQA-ID (class A/B/C, Indonesian). Tidak ada dataset native Indonesian dengan pertanyaan multi-hop (class C) yang dimasukkan ke training — karena tidak ada dataset semacam itu yang tersedia dan translation augmentation secara eksplisit ditolak (lihat catatan di bawah). Risiko ini dimitigasi melalui **post-hoc consistency test**: jalankan classifier pada XQuAD-ID eval split dengan query EN dan query ID yang identik secara semantik, bandingkan prediksi per-pair. Hasil dilaporkan sebagai tabel konsistensi di paper — apapun hasilnya memperkuat narasi: konsistensi tinggi → classifier generalizes; konsistensi rendah → confirms linguistic bias yang kemudian dijustifikasi oleh arsitektur translation-free. Limitation ini ditulis eksplisit di bagian Limitations paper.

**Mengapa tidak ada translation augmentation:**
Translation augmentation (misalnya: translate HotpotQA EN ke ID untuk menambah class C dalam Bahasa Indonesia) ditolak karena kontradiksi langsung dengan klaim arsitektur translation-free. Jika kita menggunakan translated data untuk training, reviewer dapat mempertanyakan apakah perbaikan sistem berasal dari BGE-M3 atau dari translated training signal. Konsistensi arsitektural lebih penting dari coverage training data yang marginal.

#### Retrieval Layer

| Strategy                  | Model / Config               | Role                          | Cross-lingual Support         |
| ------------------------- | ---------------------------- | ----------------------------- | ----------------------------- |
| Sparse baseline           | BM25 + Sastrawi stemmer      | Ablation — replikasi baseline | ❌ ID queries only            |
| Dense                     | BGE-M3 Dense mode            | Main upgrade                  | ✅ 100+ languages             |
| **BGE-M3 Sparse + Dense** | **BGE-M3 (both modes, RRF)** | **Recommended config**        | **✅ fully cross-lingual**    |
| Hybrid BM25 + BGE-M3      | BM25 + BGE-M3 Dense, RRF     | Ablation only                 | ⚠️ asymmetric (lihat catatan) |

**BGE-M3 internal modes** (semua dalam satu model, BAAI/bge-m3):

- **Dense mode**: embedding similarity — semantic, cross-lingual
- **Sparse mode**: learned sparse weights — cross-lingual BM25 equivalent
- **ColBERT mode**: token-level late interaction — opsional, high recall tapi mahal

**⚠️ Hybrid BM25 + BGE-M3 asymmetry:**

```
Query ID → Corpus ID:  BM25 ✅ + BGE-M3 ✅ → hybrid beneficial
Query EN → Corpus ID:  BM25 ❌ (noise) + BGE-M3 ✅ → BM25 degrades results
```

Ini empirically demonstrable dan akan dilaporkan di Δgap table sebagai bukti bahwa BM25 _memperlebar_ language gap.

#### Reranker

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384` (mMiniLM)
- Fungsi: menilai relevansi pasangan (query, dokumen) secara language-agnostic
- Diterapkan setelah top-k retrieval, sebelum generation
- **Bukan confidence gate** — mMiniLM adalah reranker retrieval, bukan mekanisme evaluasi kepercayaan jawaban LLM

#### Adaptive Generation — CRAG Multi-hop

| Complexity  | Strategy        | Keterangan                                                 |
| ----------- | --------------- | ---------------------------------------------------------- |
| A (simple)  | No retrieval    | Sama dengan baseline                                       |
| B (medium)  | Single-shot RAG | Sama dengan baseline                                       |
| C (complex) | CRAG            | Confidence-gated; context managed via replace + carry-over |

**Mekanisme CRAG (berbeda dari IRCoT baseline):**

| Aspek              | IRCoT Baseline                          | CRAG (Sistem Kita)                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Termination        | Keyword "Jawaban" OR max 5 iterasi      | LLM mengevaluasi confidence jawaban sendiri sebelum retrieve lagi                                                                                                                                                                          |
| Context management | Akumulasi semua dokumen tanpa filtering | Replace: dokumen baru menggantikan lama; carry-over 1 iterasi untuk dokumen yang masih relevan                                                                                                                                             |
| Relevance filter   | Tidak ada                               | LLM self-assessment: setelah "not confident", LLM diberi constrained prompt untuk mengidentifikasi dokumen mana dari iterasi sebelumnya yang masih relevan terhadap pertanyaan akhir — dokumen yang di-flag di-carry-over, sisanya dibuang |
| Stopping           | Tidak berbasis kualitas retrieval       | Berbasis confidence — LLM memutuskan retrieve lagi / jawab / abstain                                                                                                                                                                       |

**Mekanisme carry-over (committed):**
Setelah LLM menyatakan "not confident", pipeline menjalankan constrained prompt terpisah:
_"Dari dokumen berikut [daftar docs iterasi n-1], mana yang masih relevan untuk menjawab pertanyaan utama? Jawab hanya dengan nomor dokumen."_
Dokumen yang di-flag di-carry-over ke iterasi berikutnya. Dokumen yang tidak di-flag dibuang. Implementasi eksplisit di `05_generation_pipeline.ipynb`.

**Mengapa ini lebih baik dari IRCoT:** IRCoT mengakumulasi semua dokumen dari setiap iterasi — context window membengkak dan LLM halusinasi. CRAG mempertahankan hanya dokumen relevan melalui carry-over terbatas, mencegah akumulasi noise.

#### Complexity Classifier (Sistem Kita)

- Model: **mDeBERTa-v3-base** (`microsoft/mdeberta-v3-base`)
- Classification head: `Linear(384, 3)` — hidden size 384 sesuai arsitektur mDeBERTa-v3-base
- Training data:
  - **HotpotQA**: ~10.000 instances, semua label "C" (hardcoded, tidak melalui annotation pipeline — multi-hop by design)
  - **TyDiQA-ID**: 4.815 instances (80% stratified split dari TyDiQA-ID train), label A/B/C melalui labeling pipeline berbasis answer field:
    - **Label A** (Simple factoid): answer word count ≤ 3 kata AND jawaban berupa named entity atau angka
    - **Label B** (Medium): answer word count > 3 kata AND jawaban berupa single span phrase/clause
    - **Discarded**: instance yang tidak memenuhi kondisi A maupun B — jumlah di-log untuk paper. TyDiQA-ID tidak menghasilkan Class C dari pipeline ini.
  - **XQuAD-ID**: **tidak digunakan untuk training** — full 1.190 reserved untuk evaluasi
- **Class imbalance**: Class C berasal exclusively dari HotpotQA (~10.000); Class A dan B exclusively dari TyDiQA-ID. Severe imbalance expected.
- **Weighted loss**: wajib — `CrossEntropyLoss(weight=class_weights)` dimana `w_i = total / (n_classes × n_i)`. Tanpa weighted loss, model collapse ke Class C. Weights diterapkan di loss function, bukan di sampling.
- **Internal train/val split**: setelah merge HotpotQA + TyDiQA-ID, split 80/20 stratified **by class AND language** — memastikan Class A dan B muncul di val set. Split ini adalah split internal fine-tuning, terpisah dari split dataset untuk fase-fase evaluasi.
- **Target accuracy**: ≥ 0.756 — setara dengan IndoBERT baseline untuk memvalidasi bahwa mDeBERTa-v3-base pada distribusi yang lebih luas tidak regresi. Target ini juga menjadi gate sebelum post-hoc consistency test.
- Mengapa mDeBERTa-v3-base (bukan IndoBERT): multilingual, mendukung query EN dan ID natively, lebih cocok untuk sistem cross-lingual. IndoBERT baseline hanya valid untuk distribusi IndoQA/QASiNa (monolingual Indonesian).
- **Post-hoc consistency test**: setelah training, jalankan classifier pada 1.190 XQuAD-ID eval pairs — bandingkan prediksi untuk EN query vs ID query yang identik. Hasilnya dilaporkan di paper sebagai tabel. Lihat "Acknowledged limitation" di bagian Dataset di atas.

**Catatan untuk Baseline Reproduction (Phase 2):**
Saat mereplikasi baseline, kita hanya menggunakan **HotpotQA eval split**. HotpotQA tidak memerlukan classifier — semua pertanyaan langsung di-route ke method C (multi-retrieval), sebagaimana dinyatakan secara eksplisit oleh baseline paper. Hasil HotpotQA dipilih karena merupakan kasus kegagalan paling ekstrem dari baseline (multi-retrieval collapse ke 0.007) dan menjadi titik referensi utama untuk mengukur perbaikan di ablation study.

#### Language Model

- **Phase 2 & 3 (Baseline + Ablation)**: Qwen 3-8B — sama dengan baseline, untuk isolasi kontribusi retrieval
- **Phase 5 (RAGAS Evaluation)**: kedua model dievaluasi secara paralel untuk menghasilkan perbandingan reasoning vs. non-reasoning pada sistem yang sama

**Expected RAGAS Output Table (Phase 5):**

| System                       | LLM                | Dataset         | Faithful. | Factual Cor. | Sem. Sim. | RAGAS Avg |
| ---------------------------- | ------------------ | --------------- | --------- | ------------ | --------- | --------- |
| _Setiawan et al. [4] (ref.)_ | _Gemini-2.5-Flash_ | _Institutional_ | _0.9195_  | _0.6274_     | _0.9151_  | _0.8207_  |
| Full System                  | Qwen3-8B           | TyDiQA-ID       | [NB8]     | [NB8]        | [NB8]     | [NB8]     |
| Full System                  | Qwen3-8B           | XQuAD-ID (ID)   | [NB8]     | [NB8]        | [NB8]     | [NB8]     |
| Full System                  | Qwen3-8B           | XQuAD-ID (EN)   | [NB8]     | [NB8]        | [NB8]     | [NB8]     |
| Full System                  | Gemini-2.5-Flash   | TyDiQA-ID       | [NB8]     | [NB8]        | [NB8]     | [NB8]     |
| Full System                  | Gemini-2.5-Flash   | XQuAD-ID (ID)   | [NB8]     | [NB8]        | [NB8]     | [NB8]     |
| Full System                  | Gemini-2.5-Flash   | XQuAD-ID (EN)   | [NB8]     | [NB8]        | [NB8]     | [NB8]     |

_[NB8] = hasil dari notebook 08_ragas_evaluation. Row Setiawan et al. sebagai referensi eksternal._

---

## 4. Novel Contributions

### Contribution 1 — Quantified Retrieval Gap (Δgap)

Bangun parallel bilingual query set menggunakan XQuAD-ID (pertanyaan identik dalam ID dan EN). Ukur Recall@10 dan MRR secara terpisah per bahasa. NDCG@10 tidak dilaporkan — XQuAD-ID hanya memiliki binary relevance labels, sehingga NDCG dan R@10 memberikan sinyal yang redundan. Laporkan Δgap = metric(EN) − metric(ID) untuk setiap retrieval strategy. Reranker impact diukur terpisah menggunakan P@1 dan P@3 pada XQuAD-ID (dua baris: BGE-M3 S+D tanpa reranker vs. dengan mMiniLM). Tidak ada satupun dari ketiga paper yang mengukur ini.

### Contribution 2 — Translation-Free Cross-lingual Retrieval

Eliminasi translation step sepenuhnya dengan mengganti BM25 + OPUS-MT menggunakan BGE-M3 (sparse + dense via RRF). BGE-M3 mengenkode query ID dan EN langsung ke shared multilingual embedding space — tidak ada intermediate translation, tidak ada error propagation. Lebih kuat secara arsitektur dibanding sekadar upgrade OPUS-MT ke NLLB-200 karena menghilangkan seluruh komponen yang rawan gagal. Ablation config Hybrid BM25+BGE-M3 secara empiris menunjukkan bahwa BM25 _memperlebar_ gap pada EN queries.

### Contribution 3 — Confidence-Gated Multi-hop (CRAG)

Ganti IRCoT (termination berbasis keyword/iterasi-count) dengan CRAG yang menggunakan confidence evaluation dari LLM sendiri + context management replace-with-carry-over. Baseline paper secara eksplisit mengidentifikasi ini sebagai kegagalan terbesar dan future work.

### Contribution 4 — Cross-lingual Reranker (mMiniLM)

Tambahkan mMiniLM setelah retrieval, sebelum generation. Baseline tidak memiliki reranker sama sekali. Dalam cross-lingual setting, embedding cosine similarity tidak terkalibrasi antar bahasa — reranker mengoreksi ini dengan menilai relevansi query-dokumen secara langsung.

### Contribution 5 — Multilingual Complexity Classifier (mDeBERTa-v3-base)

Ganti IndoBERT (monolingual, trained hanya pada IndoQA/QASiNa) dengan mDeBERTa-v3-base yang di-fine-tune pada HotpotQA (~10.000 class C) + TyDiQA-ID (4.815 class A/B/C) menggunakan weighted loss untuk kompensasi class imbalance. XQuAD-ID sengaja tidak dimasukkan ke training — full 1.190 pairs direserve untuk evaluasi Δgap dan post-hoc consistency test. Limitation diakui secara eksplisit: tidak ada native Indonesian class C data di training. Ini dimitigasi melalui consistency test (EN vs ID predictions pada XQuAD-ID eval pairs) yang hasilnya dilaporkan di paper — bukan disembunyikan.

### Contribution 6 — Reasoning Model Integration

Evaluasi Gemini-2.5-Flash sebagai generation backbone, diperkuat oleh temuan Paper 2 (+15.95% factual correctness dari reasoning models). Bandingkan dengan model kecil yang digunakan baseline untuk isolasi kontribusi reasoning capability.

---

## 5. Evaluation Plan

### Metrik per Phase

| Phase                           | Dataset                                | Corpus                            | Metrik                                                                            |
| ------------------------------- | -------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| Phase 2 — Baseline Reproduction | HotpotQA eval only                     | HotpotQA source docs (BM25 Index) | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                            |
| Phase 3 — Ablation Study        | HotpotQA eval, XQuAD-ID eval (ID & EN) | Dataset-native docs               | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                            |
| Phase 4 — Δgap Measurement      | XQuAD-ID eval, TyDiQA-ID eval          | Wikipedia ID                      | Recall@10, MRR, Δgap per config; P@1, P@3 for reranker comparison (XQuAD-ID only) |
| Phase 5 — RAGAS Evaluation      | XQuAD-ID eval, TyDiQA-ID eval          | Wikipedia ID                      | Faithfulness, Factual Correctness, Semantic Similarity                            |

### Ablation Study Design (Phase 3)

| Config                | BM25+Sastrawi | BGE-M3 Dense | BGE-M3 Sparse | mMiniLM Reranker |    Reasoning LLM     | Generation  | Notes                                              |
| --------------------- | :-----------: | :----------: | :-----------: | :--------------: | :------------------: | :---------: | -------------------------------------------------- |
| Baseline              |       ✓       |      —       |       —       |        —         |          —           |    IRCoT    | Replikasi baseline (HotpotQA all-C, no classifier) |
| +Dense                |       —       |      ✓       |       —       |        —         |          —           | Single-shot | Isolasi dense retrieval gain                       |
| +Hybrid (BM25+BGE-M3) |       ✓       |      ✓       |       —       |        —         |          —           | Single-shot | Demonstrasi BM25 asymmetry pada EN queries         |
| +BGE-M3 Sparse+Dense  |       —       |      ✓       |       ✓       |        —         |          —           | Single-shot | Recommended retrieval config                       |
| +Reranker             |       —       |      ✓       |       ✓       |        ✓         |          —           | Single-shot | Isolasi kontribusi mMiniLM reranker                |
| +CRAG                 |       —       |      ✓       |       ✓       |        ✓         |          —           |    CRAG     | Isolasi perbaikan multi-hop                        |
| **Full System**       |       —       |      ✓       |       ✓       |        ✓         | ✓ (Gemini-2.5-Flash) |    CRAG     | Semua komponen                                     |

_Single-shot = standard RAG one retrieval pass. CRAG = confidence-gated multi-hop, replace-with-carry-over, max 3 iterations. Ablation configs +Dense through +CRAG use Qwen3-8B; Full System uses Gemini-2.5-Flash._

**Key comparison:** "+Hybrid" vs "+BGE-M3 Sparse+Dense" → bukti empiris BM25 memperlebar Δgap pada EN queries. Ini adalah justifikasi arsitektur translation-free.

**Locked ablation results table structure (paper V-B):** HotpotQA EM, HotpotQA F1, XQuAD F1 (ID), Δ XQuAD F1 (ID), XQuAD F1 (EN), Δ XQuAD F1 (EN). Avg F1 column dropped — misleading given HotpotQA near-zero baseline. HotpotQA delta columns omitted — near-zero baseline makes delta values uninformative.

### Accuracy Milestone Targets

| Milestone              | Target    | Komponen                       |
| ---------------------- | --------- | ------------------------------ |
| Beat baseline (QASiNa) | F1 > 0.41 | Dense retrieval saja           |
| Cross 0.5              | F1 > 0.50 | Dense + mMiniLM Reranker       |
| Cross 0.65             | F1 > 0.65 | + CRAG multi-hop               |
| Cross 0.75             | F1 > 0.75 | + Reasoning LLM                |
| Target ceiling         | F1 ≥ 0.80 | Full system (RAGAS-compatible) |

---

## 6. Paper Structure (IEEE Format)

| Section          | Focus                                                              | Key Content                                                                                                                                                                                                                                                                  | Notes                                                            |
| ---------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Abstract**     | Gap quantification + improvement over baseline                     | Language gap, 5 contributions, improvement vs. 2510.21068                                                                                                                                                                                                                    | 150–250 words, lead dengan gap problem                           |
| **Introduction** | Motivasi dengan angka kegagalan konkret                            | Baseline 0.007 accuracy HotpotQA; Paper 2 +15.95% reasoning models; classifier validity gap                                                                                                                                                                                  | End dengan 5 numbered contributions                              |
| **Related Work** | Posisi terhadap tiga stream                                        | Adaptive RAG, cross-lingual retrieval, Indonesian NLP                                                                                                                                                                                                                        | Distinguish dari ketiga reviewed papers                          |
| **Method**       | Full pipeline                                                      | (a) Bilingual query processor + mDeBERTa classifier, (b) BGE-M3 retrieval + mMiniLM reranker, (c) CRAG confidence-gated multi-hop, (d) Δgap quantification                                                                                                                   | Fokus pada novelty dan reproducibility                           |
| **Experiments**  | Setup, dataset, model, ablation config                             | Tabel ablation di atas; dataset split, hardware, hyperparameters                                                                                                                                                                                                             | Must be reproducible                                             |
| **Results**      | Δgap table + ablation table + RAGAS + perbandingan dengan baseline | V-A: Table III.A (R@10, MRR per strategy) + Table III.B (P@1, P@3 reranker, XQuAD-ID only); V-B: End-to-end ablation (HotpotQA EM/F1, XQuAD F1 ID/EN, per-dataset deltas); V-C: RAGAS (Qwen3-8B + Gemini-2.5-Flash, TyDiQA-ID + XQuAD-ID ID/EN); V-D: Classifier performance | Tabel + bar chart; lead with Δgap insight                        |
| **Conclusion**   | Kontribusi dan arah masa depan                                     | Δgap sebagai reusable benchmark; future: fine-tuned Indonesian reasoning LLM                                                                                                                                                                                                 | Rekomendasikan XQuAD-ID sebagai standard cross-lingual benchmark |

---

## 7. Component-to-Paper Mapping

| Komponen                      | Mengatasi                                                        | Bukti dari Paper                                                                                                                                                                                                              |
| ----------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BGE-M3 sparse + dense (RRF)   | BM25 sparse failure, translation bottleneck, language asymmetry  | Paper 1: BM25 → poor cross-lingual recall; Paper 2: dense enables multilingual stability                                                                                                                                      |
| Translation-free architecture | OPUS-MT semantic drift — dieliminasi sepenuhnya                  | Paper 1: admits translation drift; menghapus translation step lebih kuat dari upgrade ke NLLB-200                                                                                                                             |
| BM25 + Sastrawi (ablation)    | Baseline replication + asymmetry demonstration                   | Paper 1: BM25 digunakan naively; Sastrawi adalah minimum viable ID setup untuk fair comparison                                                                                                                                |
| mMiniLM reranker              | Uncalibrated cross-lingual embeddings                            | Paper 1: no reranker; Paper 2: retrieval quality is dominant factor                                                                                                                                                           |
| CRAG confidence-gated         | IRCoT termination lemah; akumulasi context tanpa filtering       | Paper 1: multi-retrieval collapse 7.8×–14× tergantung dataset; explicitly listed as future work                                                                                                                               |
| mDeBERTa-v3-base classifier   | IndoBERT classifier tidak tervalidasi pada HotpotQA; monolingual | Paper 1: classifier 0.756 valid hanya untuk IndoQA/QASiNa distribution. Sistem kita: trained pada HotpotQA (class C) + TyDiQA-ID (class A/B/C) dengan weighted loss; XQuAD-ID reserved untuk eval + post-hoc consistency test |
| Reasoning LLM                 | Halusinasi di bawah konteks Indonesian panjang                   | Paper 2: +15.95% factual correctness dari reasoning models                                                                                                                                                                    |
| Δgap measurement (per config) | Tidak ada language gap quantification di baseline                | Tidak ada dari ketiga paper yang mengukur ini; BM25 hybrid ablation membuktikan asymmetry secara empiris                                                                                                                      |
| RAGAS evaluation              | EM/F1 saja tidak cukup untuk grounding                           | Paper 2: RAGAS memberikan faithfulness + factual correctness                                                                                                                                                                  |
| Wikipedia ID corpus           | id_newspapers_2018 kurang defensible secara akademik             | Pengganti yang lebih umum dipakai sebagai benchmark corpus dalam NLP research                                                                                                                                                 |
| XQuAD-ID + TyDiQA-ID          | Noisy translated training data dari baseline                     | Paper 1: admits OPUS-MT drift; native datasets eliminasi ini                                                                                                                                                                  |

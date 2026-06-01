# Notebook Plan: Cross-lingual RAG for Indonesian–English QA

> Mengikuti 4-phase experiment structure dari revisi research plan. Setiap notebook dirancang bisa dijalankan independen dengan checkpoint I/O yang jelas.

---

## ⚙️ Kaggle + T4 GPU Environment — Global Notes

Seluruh pipeline ini dirancang untuk berjalan di **Kaggle Notebooks** dengan akselerator **NVIDIA Tesla T4 (16GB VRAM)**. Beberapa constraint dan workaround yang berlaku secara global:

### Hardware & Quota

- **VRAM:** 16GB — cukup untuk Qwen3-8B (4-bit quantized via bitsandbytes), BGE-M3 (fp16), dan mMiniLM secara terpisah. **Jangan load dua model besar sekaligus** tanpa `del` + `torch.cuda.empty_cache()` di antara keduanya.
- **GPU quota:** ~30 jam/minggu (Kaggle free tier). Notebook yang lama (NB02 classifier training, NB06 ablation 7 config) harus di-schedule dengan sadar — jangan run semua sekaligus di hari yang sama.
- **Session timeout:** Kaggle akan disconnect idle session setelah ~1 jam tanpa output. Pastikan setiap cell panjang mencetak progress log atau gunakan `tqdm` agar sesi tetap aktif.
- **Disk:** `/kaggle/working/` = 20GB sementara (hilang saat sesi berakhir). Gunakan strategi persist output di bawah.

### Secrets & API Keys

Simpan semua credentials via **Kaggle Secrets** (Settings → Add-ons → Secrets), bukan hardcode:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
HF_TOKEN       = secrets.get_secret("HF_TOKEN")        # untuk HuggingFace Hub
GEMINI_API_KEY = secrets.get_secret("GEMINI_API_KEY")  # untuk NB08
```

Secret names yang disarankan: `HF_TOKEN`, `GEMINI_API_KEY`.

### Persisting Outputs Across Sessions (PENTING)

Karena `/kaggle/working/` bersifat ephemeral, gunakan strategi berikut untuk semua output penting:

**Opsi A — Kaggle Dataset (direkomendasikan):**

```python
import os
os.makedirs("/kaggle/working/outputs", exist_ok=True)
# ... simpan file ke /kaggle/working/outputs/ ...
# Di akhir notebook, commit via Kaggle API atau UI "Save & Run All"
```

Lalu di notebook berikutnya, attach dataset tersebut sebagai input via "Add Data".

**Opsi B — HuggingFace Hub:**
Push model checkpoint (NB02) dan index artifacts langsung ke HF private repo. Lihat catatan per-notebook.

**Opsi C — Google Drive:**
Alternatif terakhir jika Kaggle Dataset quota habis, mount via `gdown` atau OAuth.

Gunakan **tiga Kaggle Datasets** sebagai persistent storage antar notebook:

| Dataset Name               | Isi                                                    | Dipakai oleh     |
| -------------------------- | ------------------------------------------------------ | ---------------- |
| `crosslingual-rag-data`    | Semua JSONL splits + Wikipedia corpus (output NB01)    | NB02–NB08        |
| `crosslingual-rag-indexes` | BM25 pickles + FAISS indexes (output NB03)             | NB04, NB06–NB08  |
| `crosslingual-rag-results` | Semua JSON/CSV hasil evaluasi (output NB02, NB04–NB08) | NB06–NB08, paper |

Di awal setiap notebook, attach dataset yang relevan sebagai input dan akses via `/kaggle/input/<dataset-name>/`.

### Instalasi Package

Beberapa package tidak tersedia by default di Kaggle. Tambahkan cell instalasi ini di awal setiap notebook yang memerlukannya:

```python
!pip install -q rank_bm25 PySastrawi langdetect lingua-language-detector \
             faiss-cpu FlagEmbedding sentence-transformers transformers \
             accelerate bitsandbytes ragas datasets huggingface_hub
```

> ⚠️ `bitsandbytes` untuk quantization Qwen3-8B — wajib untuk NB05, NB06, NB08.

### fp16 Everywhere

Selalu gunakan `use_fp16=True` atau `torch_dtype=torch.float16` untuk semua model yang mendukungnya di T4:

```python
# BGE-M3
from FlagEmbedding import FlagModel
model = FlagModel('BAAI/bge-m3', use_fp16=True)

# mMiniLM reranker
from FlagEmbedding import FlagReranker
reranker = FlagReranker('cross-encoder/mmarco-mMiniLMv2-L12-H384', use_fp16=True)

# Transformer models
from transformers import AutoModel
model = AutoModel.from_pretrained(..., torch_dtype=torch.float16)
```

### Qwen3-8B Loading di T4

Qwen3-8B full-precision tidak muat di 16GB VRAM. Gunakan 4-bit quantization di semua notebook yang memerlukannya:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

### FAISS di Kaggle

FAISS index disimpan dan di-load via file `.index` + metadata pickle. Selalu set path eksplisit ke direktori writable:

```python
import faiss, os
os.makedirs("/kaggle/working/faiss_db", exist_ok=True)

# Simpan index setelah build
faiss.write_index(index, "/kaggle/working/faiss_db/wikipedia_id.index")

# Load index di session berikutnya
index = faiss.read_index("/kaggle/working/faiss_db/wikipedia_id.index")
```

> ⚠️ FAISS index hilang saat sesi berakhir — **wajib** zip dan simpan ke Kaggle Dataset `crosslingual-rag-indexes` sebelum session end:

```python
import shutil
shutil.make_archive(
    "/kaggle/working/faiss_bge_m3_wikipedia_id",
    "zip",
    "/kaggle/working/faiss_db"
)
# Upload zip ke Kaggle Dataset via UI atau API
```

Di notebook berikutnya yang butuh index ini, unzip di awal session:

```python
shutil.unpack_archive(
    "/kaggle/input/crosslingual-rag-indexes/faiss_bge_m3_wikipedia_id.zip",
    "/kaggle/working/faiss_db"
)
index = faiss.read_index("/kaggle/working/faiss_db/wikipedia_id.index")
```

---

## Notebook Overview

| #   | Notebook                         | Phase          | Input                               | Output                                                                | Est. GPU Time (T4) |
| --- | -------------------------------- | -------------- | ----------------------------------- | --------------------------------------------------------------------- | :----------------: |
| 01  | `01_data_prep.ipynb`             | Pre-experiment | Raw datasets                        | Cleaned splits, retrieval corpus index                                | ~0.5 jam (CPU ok)  |
| 02  | `02_complexity_classifier.ipynb` | Pre-experiment | TyDiQA-ID, HotpotQA                 | Fine-tuned mDeBERTa-v3-base checkpoint                                |      ~3–5 jam      |
| 03  | `03_retrieval_indexing.ipynb`    | Pre-experiment | Wikipedia ID (~20K), HotpotQA docs  | BM25 index, BGE-M3 vector store (FAISS)                               |      ~2–3 jam      |
| 04  | `04_reranker.ipynb`              | Pre-experiment | Query-doc pairs                     | Reranker scores; calibration sanity check                             |       ~1 jam       |
| 05  | `05_generation_pipeline.ipynb`   | Phase 1 & 2    | Queries + retrieved docs            | Answer outputs per config per dataset                                 |       ~2 jam       |
| 06  | `06_ablation_study.ipynb`        | Phase 2        | Outputs from `05`                   | Ablation results table (EM, F1)                                       |      ~5–8 jam      |
| 07  | `07_dgap_measurement.ipynb`      | Phase 3        | XQuAD-ID queries (EN + ID parallel) | Δgap table (Recall@10, MRR); reranker precision table (P@1, P@3)      |      ~1–2 jam      |
| 08  | `08_evaluation_ragas.ipynb`      | Phase 4        | Full system outputs                 | RAGAS scores (Faithfulness, Factual Correctness, Semantic Similarity) |      ~3–4 jam      |

> 🟡 **Kaggle T4 Note — Overview:** Total estimated GPU time ~17–24 jam. Dengan kuota 30 jam/minggu, ini feasible dalam 1 minggu jika dibagi 2–3 sesi. **NB02 dan NB06 adalah yang paling berat** — prioritaskan GPU quota untuk keduanya. NB01 bisa dijalankan tanpa GPU sama sekali.

---

## Notebook Details

### `01_data_prep.ipynb`

**Tujuan:** Load, clean, dan split semua dataset. Pastikan format konsisten di seluruh pipeline.

**Tasks:**

- Load HotpotQA (eval split, 7.405 instances) + train split (~10.000 instances untuk NB02)
- Load TyDiQA-ID — pisahkan train (4.815) / eval (1.452) dengan stratified 80/20 split
- Load XQuAD-ID (1.190 full) — **pertahankan pasangan paralel EN/ID, jangan pisah**. Ini kunci untuk Δgap calculation di NB07. Setiap row harus punya `question_id`, `question_en`, `question_id_text` (pertanyaan dalam bahasa Indonesia), `answer`, `context`.
- Load **Wikipedia ID (~20.000 artikel)** — ini **retrieval corpus only** untuk NB07 (Δgap) dan NB08 (RAGAS). Bukan eval dataset.
- Load **HotpotQA source documents** — retrieval corpus untuk ablation (NB06). Bukan eval dataset.
- Export ke format unified JSON: `{id, question, answer, lang, dataset, split}`

**Output files:**

```
data/hotpotqa_train.jsonl           # ~10.000 instances untuk NB02 classifier training
data/hotpotqa_eval.jsonl            # 7.405 instances untuk ablation (NB06)
data/hotpotqa_source_docs.jsonl     # retrieval corpus untuk ablation (NB03, NB06)
data/tydiqa_id_train.jsonl          # 4.815 instances untuk NB02 classifier training
data/tydiqa_id_eval.jsonl           # 1.452 instances untuk NB07 (Δgap) dan NB08 (RAGAS)
data/xquad_id_parallel.jsonl        # 1.190 instances; kolom: question_id, question_en,
                                    # question_id_text, answer, context
                                    # JANGAN split EN dan ID ke file terpisah
data/wikipedia_id_corpus.jsonl      # retrieval corpus untuk NB07 (Δgap) dan NB08 (RAGAS)
```

**Catatan:** Jangan translate apapun. BGE-M3 di NB03 handles cross-lingual natively.

> 🟡 **Kaggle T4 Note — NB01:**
>
> - Jalankan **tanpa GPU** (set accelerator ke "None") untuk hemat quota — NB01 hanya operasi CPU.
> - Load semua dataset via HuggingFace `datasets` library:
>
>   ```python
>   from datasets import load_dataset
>
>   hotpotqa = load_dataset("hotpot_qa", "distractor")
>   tydiqa   = load_dataset("khalidalt/tydiqa-goldp", "secondary_task")
>   xquad    = load_dataset("xquad", "xquad.id")
>   # Wikipedia ID — sample 20.000 artikel:
>   wiki_id  = load_dataset("wikipedia", "20220301.id", split="train").shuffle(seed=42).select(range(20000))
>   ```
>
> - **Simpan semua output JSONL ke Kaggle Dataset `crosslingual-rag-data`** sebelum sesi berakhir. Dataset ini di-attach sebagai input di NB02–NB08.
> - Total file size output ~500MB–1GB — masuk dalam limit Kaggle Dataset (20GB).
> - Pastikan `xquad_id_parallel.jsonl` menyimpan **kedua bahasa dalam satu row** dengan key `question_id` yang sama — ini krusial untuk Δgap calculation di NB07.

---

### `02_complexity_classifier.ipynb`

**Tujuan:** Fine-tune **mDeBERTa-v3-base** sebagai multilingual complexity classifier (label: A/B/C). Ini adalah satu-satunya komponen yang benar-benar di-train. Model ini harus handle query EN dan ID secara native — IndoBERT tidak digunakan karena monolingual.

**Model:**

- `microsoft/mdeberta-v3-base`
- Classification head: `Linear(384, 3)` — hidden size 384 sesuai arsitektur mDeBERTa-v3-base
- 3-class output: A (simple/no retrieval), B (medium/single-shot), C (complex/multi-hop)

**Training Data & Labeling Pipeline:**

**HotpotQA (~10.000 instances):**

- Semua instance di-assign label **C** (hardcoded — multi-hop by design, tidak melalui annotation pipeline)
- Tidak ada filtering atau labeling heuristic untuk HotpotQA

**TyDiQA-ID (4.815 instances — 80% stratified split dari TyDiQA-ID train):**

- Label ditentukan via answer field heuristic:
  - **Label A** (Simple factoid): answer word count ≤ 3 kata AND jawaban berupa named entity atau angka
  - **Label B** (Medium): answer word count > 3 kata AND jawaban berupa single span phrase/clause
  - **Discarded**: instance yang tidak memenuhi kondisi A maupun B — jumlah di-log untuk paper
  - TyDiQA-ID tidak menghasilkan Class C dari pipeline ini
- XQuAD-ID: **tidak digunakan untuk training** — full 1.190 reserved untuk evaluasi

**Class Imbalance & Weighted Loss:**

- Class C berasal exclusively dari HotpotQA (~10.000); Class A dan B exclusively dari TyDiQA-ID → severe imbalance
- **Wajib**: `CrossEntropyLoss(weight=class_weights)` dimana `w_i = total / (n_classes × n_i)`
- Weights diterapkan di loss function, bukan di data sampling
- Tanpa weighted loss, model akan collapse ke Class C

**Internal Train/Val Split:**

- Setelah merge HotpotQA + TyDiQA-ID, split 80/20 stratified **by class AND language**
- Tujuan: memastikan Class A dan B muncul di val set
- Split ini terpisah dari dataset eval split yang digunakan di phase-phase evaluasi

**Hyperparameters (locked):**

- 5 epochs, batch size 16, max sequence length 128

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1 per class (bukan Loss — Loss hanya untuk monitor training loop, tidak dilaporkan sebagai hasil)
- **Target: ≥ 0.756 accuracy** — setara IndoBERT baseline

**Post-hoc Consistency Test (wajib dijalankan setelah training):**

- Jalankan classifier pada 1.190 XQuAD-ID eval pairs
- Bandingkan prediksi untuk EN query vs ID query yang identik secara semantik
- Hasilnya dilaporkan di paper sebagai tabel konsistensi — hasil apapun dilaporkan, tidak disembunyikan
- Jika konsistensi tinggi → classifier generalizes cross-lingually
- Jika konsistensi rendah → confirms linguistic bias, dijustifikasi oleh arsitektur translation-free

**Output files:**

```
models/complexity_classifier/        # saved mDeBERTa-v3-base checkpoint
results/classifier_eval.json         # accuracy, precision, recall, F1 per class
results/classifier_consistency.json  # per-pair EN vs ID prediction consistency pada XQuAD-ID eval
```

> 🟡 **Kaggle T4 Note — NB02:**
>
> - mDeBERTa-v3-base dengan `max_length=128` dan `batch_size=16` muat di T4 tanpa masalah. **Jangan naikkan max_length ke 512** — tidak perlu untuk task ini dan akan memperlambat training secara signifikan.
> - Estimasi training time: **~3–5 jam** untuk 5 epoch pada ~14.000 instances. Gunakan `tqdm` per-batch agar sesi tidak timeout.
> - **Push checkpoint ke HuggingFace Hub segera setelah training selesai** — jangan tunggu sesi mati:
>
>   ```python
>   from huggingface_hub import login
>   login(token=HF_TOKEN)
>   model.push_to_hub("username/mdeberta-complexity-id", private=True)
>   tokenizer.push_to_hub("username/mdeberta-complexity-id", private=True)
>   ```
>
> - Simpan `classifier_eval.json` dan `classifier_consistency.json` ke Kaggle Dataset `crosslingual-rag-results`.
> - Gunakan `torch.cuda.empty_cache()` setelah training sebelum menjalankan consistency test agar VRAM tidak penuh.
> - Input notebook ini: attach `crosslingual-rag-data` untuk mengakses `hotpotqa_train.jsonl` dan `tydiqa_id_train.jsonl`.

---

### `03_retrieval_indexing.ipynb`

**Tujuan:** Build semua retrieval indexes yang diperlukan untuk ablation dan Δgap measurement. **Δgap hanya diukur di XQuAD-ID** — notebook ini mempersiapkan semua index yang dibutuhkan.

**Dua corpus yang diindex secara terpisah:**

- **HotpotQA source docs** → untuk ablation (NB06), dataset-native corpus
- **Wikipedia ID (~20.000 artikel)** → untuk Δgap measurement (NB07) dan RAGAS evaluation (NB08), open-domain corpus

**Alasan dua corpus:** Ablation menggunakan dataset-native docs untuk mengisolasi kontribusi komponen. Δgap dan RAGAS menggunakan Wikipedia ID karena dataset-native passages sudah semantically aligned ke pertanyaannya — ini akan artificially inflate retrieval scores dan menyembunyikan language gap yang sebenarnya.

**Tasks:**

1. **BM25 index** (untuk ablation baseline config):
   - Sastrawi stemmer + stopword removal untuk dokumen Indonesia
   - Index **HotpotQA source docs** dan **Wikipedia ID** secara terpisah
   - Simpan sebagai `rank_bm25` pickle (satu file per corpus)

2. **BGE-M3 Dense index** (FAISS):
   - Model: `BAAI/bge-m3` (use_fp16=True)
   - Embed **HotpotQA source docs** dan **Wikipedia ID** secara terpisah ke dua index FAISS terpisah
   - Cosine similarity

3. **BGE-M3 Sparse index**:
   - Gunakan BGE-M3 sparse mode (learned sparse weights)
   - Fuse dengan dense via RRF (k=60) untuk konfigurasi "Sparse+Dense"
   - Index **HotpotQA source docs** dan **Wikipedia ID** secara terpisah

4. **Sanity check cross-lingual retrieval:**
   - Ambil 10 sampel dari XQuAD-ID (5 EN, 5 ID versi paralel)
   - Query ke Wikipedia ID corpus
   - Bandingkan top-5 docs dari BM25 vs BGE-M3 Dense untuk query yang sama
   - Log: apakah BM25 benar-benar gagal di EN queries terhadap corpus ID?

**Output files:**

```
indexes/bm25_hotpotqa.pkl
indexes/bm25_wikipedia_id.pkl
indexes/faiss_bge_m3_hotpotqa/
indexes/faiss_bge_m3_wikipedia_id/
results/retrieval_sanity_check.json
```

> 🟡 **Kaggle T4 Note — NB03:**
>
> - BGE-M3 embedding ~20.000 Wikipedia artikel adalah operasi **paling berat** di notebook ini. Estimasi: ~2–3 jam. Gunakan `use_fp16=True` dan batch size 32–64:
>
>   ```python
>   from FlagEmbedding import FlagModel
>   model = FlagModel('BAAI/bge-m3', use_fp16=True)
>   # Embed in batches
>   embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)
>   ```
>
> - **FAISS index wajib di-persist ke Kaggle Dataset `crosslingual-rag-indexes`** — ini artifact terbesar dan paling mahal untuk di-rebuild. Zip setelah indexing selesai (lihat contoh kode di Global Notes).
> - BM25 pickle relatif kecil (<100MB per corpus) — simpan ke dataset yang sama.
> - Di NB06, NB07, NB08: unzip FAISS di awal session sebelum load FAISS index.
> - Jika VRAM penuh saat embed (unlikely tapi mungkin): kurangi batch size ke 16.
> - Input notebook ini: attach `crosslingual-rag-data` untuk mengakses `hotpotqa_source_docs.jsonl` dan `wikipedia_id_corpus.jsonl`.

---

### `04_reranker.ipynb`

**Tujuan:** Setup, validasi, dan benchmark cross-lingual reranker. Output notebook ini mengisi **paper Table III.B (Reranker Precision on XQuAD-ID)**.

**Model (locked):**

- `cross-encoder/mmarco-mMiniLMv2-L12-H384` (mMiniLM) — use_fp16=True
- Tidak ada alternatif model; BAAI/bge-reranker-v2-m3 tidak digunakan

**Fungsi reranker:** Menilai relevansi pasangan (query, dokumen) secara language-agnostic. Diterapkan setelah top-20 retrieval, sebelum generation. Ini adalah **retrieval reranker only** — bukan confidence gate, bukan bagian dari CRAG loop.

**Tasks:**

1. **Validasi ranking shift:**
   - Input: XQuAD-ID parallel queries (EN dan ID) → Wikipedia ID corpus → top-20 candidates dari BGE-M3 S+D
   - Jalankan mMiniLM reranking, ambil top-5
   - Bandingkan ranking sebelum dan sesudah reranking untuk EN vs ID queries

2. **Precision measurement untuk Table III.B:**
   - Hitung **P@1** dan **P@3** untuk dua kondisi: BGE-M3 S+D tanpa reranker vs. BGE-M3 S+D + mMiniLM
   - Dijalankan pada **XQuAD-ID saja**
   - Pisahkan hasil EN queries dan ID queries, hitung ΔP@1 dan ΔP@3

3. **Latency benchmark:**
   - Mean reranking time per query (dilaporkan di paper sebagai efficiency note)

**Output files:**

```
models/reranker/
results/reranker_validation.json
results/reranker_precision_table.json   # → feeds paper Table III.B
```

**Paper Table III.B yang diisi notebook ini:**

```
| Configuration        | P@1 EN | P@1 ID | ΔP@1 | P@3 EN | P@3 ID | ΔP@3 |
| BGE-M3 S+D           | [NB4]  | [NB4]  | [NB4]| [NB4]  | [NB4]  | [NB4]|
| BGE-M3 S+D + mMiniLM | [NB4]  | [NB4]  | [NB4]| [NB4]  | [NB4]  | [NB4]|
```

> 🟡 **Kaggle T4 Note — NB04:**
>
> - mMiniLM reranker relatif ringan (384-dim). Dengan `use_fp16=True`, VRAM usage ~1–2GB — bisa co-exist dengan BGE-M3. Namun untuk safety, load BGE-M3 → get top-20 candidates → `del model; torch.cuda.empty_cache()` → load reranker.
> - NB04 bergantung pada FAISS index dari NB03 — **unzip index di awal session** sebelum load FAISS index.
> - Estimasi runtime: ~1 jam untuk 1.190 XQuAD-ID pairs × top-20 candidates.
> - Simpan `reranker_precision_table.json` ke Kaggle Dataset `crosslingual-rag-results` — NB07 akan load file ini langsung dari sana tanpa menghitung ulang.
> - Input notebook ini: attach `crosslingual-rag-data` (untuk XQuAD-ID) dan `crosslingual-rag-indexes` (untuk FAISS).

---

### `05_generation_pipeline.ipynb`

**Tujuan:** Implement adaptive generation (A/B/C) dan jalankan **Phase 1 (baseline reproduction)**.

**Tasks:**

**Strategy implementation:**

- Strategy A: No retrieval — langsung ke LLM
- Strategy B: Single-shot RAG — retrieve top-k → LLM
- Strategy C: CRAG / Self-RAG — confidence-gated loop (bukan blind IRCoT loop)

**Phase 1 — Baseline Reproduction:**

- Config: BM25 + Qwen3-8B + IRCoT (tanpa reranker, tanpa CRAG)
- Dataset: **HotpotQA saja** di Phase 1
- Target output: F1 ≈ 0.059 ± 0.03 (validasi reproduksi)

**Phase 1 extension untuk baseline reference points:**

- Jalankan baseline config yang sama di **TyDiQA-ID dan XQuAD-ID**
- Framing di paper: _"We run the baseline configuration on TyDiQA-ID and XQuAD-ID to establish reference points, although the original paper did not evaluate on these datasets."_

**Output files:**

```
results/phase1/baseline_hotpotqa.jsonl
results/phase1/baseline_tydiqa.jsonl
results/phase1/baseline_xquad.jsonl
```

> 🟡 **Kaggle T4 Note — NB05:**
>
> - **Load Qwen3-8B dengan 4-bit quantization** (wajib di T4 16GB) — lihat template di Global Notes.
> - IRCoT loop dengan max-5 iterasi pada 7.405 HotpotQA instances bisa memakan waktu lama. **Jalankan dulu sample 200 instances** untuk validasi F1 ≈ 0.059 sebelum full run.
> - Simpan output jsonl **secara incremental** (setiap 500 instances) ke Kaggle Dataset — jangan tunggu full run selesai:
>
>   ```python
>   # Setiap 500 instances, append ke file output
>   with open("/kaggle/working/baseline_hotpotqa.jsonl", "a") as f:
>       for result in batch_results:
>           f.write(json.dumps(result) + "\n")
>   ```
>
> - CRAG implementation (Strategy C) di notebook ini harus di-test pada **≤50 instances** dulu sebelum dipakai di NB06.
> - Input notebook ini: attach `crosslingual-rag-data` dan `crosslingual-rag-indexes`.

---

### `06_ablation_study.ipynb`

**Tujuan:** Jalankan semua 7 konfigurasi pada dataset yang benar (Phase 2). **Semua baris ablation table harus terisi — tidak boleh ada missing cells.** Output notebook ini mengisi **paper Table V (End-to-End Ablation)**.

**Dataset yang digunakan:**

- **HotpotQA eval** (7.405 instances) — EN multi-hop, corpus: HotpotQA source docs
- **XQuAD-ID eval** (1.190 instances) — dijalankan dua kali: sekali dengan query EN, sekali dengan query ID, corpus: XQuAD-ID passages (dataset-native)
- **TyDiQA-ID: TIDAK digunakan di ablation** — TyDiQA-ID train split digunakan untuk classifier training (NB02), dan eval split direserve untuk Δgap (NB07) dan RAGAS (NB08). Memasukkan TyDiQA-ID ke ablation membuka risiko distributional familiarity karena classifier sudah di-train pada distribusi yang sama.

**Corpus:** Dataset-native untuk semua config di notebook ini (bukan Wikipedia ID — itu hanya untuk NB07 dan NB08).

**Konfigurasi (locked):**

| Config               | BM25+Sastrawi | BGE-M3 Dense | BGE-M3 Sparse | mMiniLM | Generation  |       LLM        |
| -------------------- | :-----------: | :----------: | :-----------: | :-----: | :---------: | :--------------: |
| Baseline             |       ✓       |      —       |       —       |    —    |    IRCoT    |     Qwen3-8B     |
| +Dense               |       —       |      ✓       |       —       |    —    | Single-shot |     Qwen3-8B     |
| +Hybrid BM25+BGE-M3  |       ✓       |      ✓       |       —       |    —    | Single-shot |     Qwen3-8B     |
| +BGE-M3 Sparse+Dense |       —       |      ✓       |       ✓       |    —    | Single-shot |     Qwen3-8B     |
| +mMiniLM Reranker    |       —       |      ✓       |       ✓       |    ✓    | Single-shot |     Qwen3-8B     |
| +CRAG                |       —       |      ✓       |       ✓       |    ✓    |    CRAG     |     Qwen3-8B     |
| Full System          |       —       |      ✓       |       ✓       |    ✓    |    CRAG     | Gemini-2.5-Flash |

**Generation strategies (locked definitions):**

- **IRCoT**: keyword-terminated atau max-5-iterasi loop — replikasi baseline
- **Single-shot**: standard RAG, satu retrieval pass, top-5 passages ke LLM
- **CRAG**: confidence-gated, max 3 iterasi, replace-with-carry-over context management

**Metrics per run (locked):** EM dan F1 — tidak ada Accuracy, tidak ada Loss. Hitung per-dataset delta vs Baseline row.

**Paper Table V yang diisi notebook ini:**

```
| Config               | HotpotQA EM | HotpotQA F1 | XQuAD F1 (ID) | Δ XQuAD F1 (ID) | XQuAD F1 (EN) | Δ XQuAD F1 (EN) |
| Baseline             | [NB6]       | [NB6]       | [NB6]         | —               | [NB6]         | —               |
| +Dense               | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
| +Hybrid BM25+BGE-M3  | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
| +BGE-M3 Sparse+Dense | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
| +mMiniLM Reranker    | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
| +CRAG                | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
| Full System          | [NB6]       | [NB6]       | [NB6]         | [NB6]           | [NB6]         | [NB6]           |
```

**Avg F1 tidak dihitung** — dropped karena HotpotQA near-zero baseline membuat unweighted average misleading. HotpotQA delta columns juga tidak ada — near-zero baseline membuat delta uninformative.

**Catatan kunci — "+Hybrid (BM25+BGE-M3)" config:**
Ini sengaja dijalankan untuk menunjukkan bahwa BM25 _widens_ Δgap pada EN queries. Bandingkan "+Hybrid" vs "+BGE-M3 Sparse+Dense" di XQuAD-ID (EN queries) — ini adalah bukti empiris utama untuk justifikasi arsitektur translation-free.

**Statistical significance:** Paired bootstrap resampling (n=1.000, α=0.05) antara setiap adjacent config pair. Dilaporkan di paper untuk setiap peningkatan yang diklaim signifikan.

> 🟡 **Kaggle T4 Note — NB06:**
>
> - NB06 adalah notebook **paling berat** — 7 config × 3 dataset splits × ribuan instances. Estimasi: **5–8 jam GPU**. **Bagi menjadi 2 sesi Kaggle terpisah:**
>   - **Sesi A:** Config Baseline, +Dense, +Hybrid BM25+BGE-M3
>   - **Sesi B:** +BGE-M3 S+D, +Reranker, +CRAG, Full System
> - **Simpan hasil tiap config ke file jsonl terpisah segera setelah config selesai** — jangan tunggu semua config beres:
>
>   ```python
>   # Setelah setiap config selesai:
>   import json, os
>   output_path = f"/kaggle/working/ablation_{config_name}_{dataset}_{lang}.jsonl"
>   with open(output_path, "w") as f:
>       for r in results:
>           f.write(json.dumps(r) + "\n")
>   # Upload ke Kaggle Dataset crosslingual-rag-results segera
>   ```
>
> - **Full System config menggunakan Gemini-2.5-Flash (API)** — tidak butuh GPU untuk generation, tapi masih butuh GPU untuk BGE-M3 retrieval + mMiniLM reranker. Load Gemini via API key dari Kaggle Secrets.
> - Untuk semua Qwen3-8B configs: gunakan 4-bit quantization seperti di Global Notes.
> - Setelah setiap config yang berganti model: `del model; torch.cuda.empty_cache()` sebelum load model berikutnya.
> - Statistical significance (paired bootstrap n=1.000) jalankan di **akhir setelah semua configs selesai** — tidak butuh GPU, bisa di session CPU-only.
> - Input notebook ini: attach `crosslingual-rag-data` dan `crosslingual-rag-indexes`.

---

### `07_dgap_measurement.ipynb`

**Tujuan:** Kuantifikasi language gap per arsitektur retrieval. **Hanya XQuAD-ID, corpus: Wikipedia ID.** Ini adalah Phase 3. Output notebook ini mengisi **paper Table III.A (Cross-lingual Retrieval Gap)**.

**Corpus: Wikipedia ID (~20.000 artikel)** — bukan dataset-native passages. Alasan: dataset-native passages sudah semantically aligned ke pertanyaannya sehingga artificially inflate scores. Wikipedia ID memberikan pengukuran gap yang realistis pada open-domain setting.

**Input:** XQuAD-ID parallel queries — `question_en` dan `question_id` untuk pertanyaan yang identik secara semantik, diidentifikasi via `question_id` yang sama (dijaga dari NB01).

**Tasks:**

**Table III.A — Retrieval Gap (4 strategi):**

| Config yang diukur  | R@10 EN | R@10 ID | ΔR@10 | MRR EN | MRR ID | ΔMRR |
| ------------------- | ------- | ------- | ----- | ------ | ------ | ---- |
| BM25                | ...     | ...     | ...   | ...    | ...    | ...  |
| BGE-M3 Dense        | ...     | ...     | ...   | ...    | ...    | ...  |
| BGE-M3 Sparse+Dense | ...     | ...     | ...   | ...    | ...    | ...  |
| Hybrid BM25+BGE-M3  | ...     | ...     | ...   | ...    | ...    | ...  |

**Metrics (locked):**

- **Recall@10** dan **MRR** — hanya dua metrik ini
- **NDCG@10 tidak dihitung** — XQuAD-ID hanya memiliki binary relevance labels; dengan binary relevance, NDCG dan R@10 memberikan sinyal yang redundan
- **Δgap = metric(EN) − metric(ID)** — positif = sistem bias ke EN queries

**Table III.B** — load `reranker_precision_table.json` dari NB04 dan format untuk paper. Tidak perlu menghitung ulang.

**Output files:**

```
results/phase3/dgap_table_IIIA.csv
results/phase3/dgap_table_IIIA_formatted.md
results/phase3/dgap_table_IIIB_formatted.md
```

> 🟡 **Kaggle T4 Note — NB07:**
>
> - NB07 hanya menjalankan **retrieval** (tidak ada LLM generation) — relatif ringan, ~1–2 jam total.
> - Unzip FAISS Wikipedia ID index di awal session (lihat Global Notes untuk kode unzip).
> - BM25 retrieval untuk 1.190 × 2 (EN + ID) queries sangat cepat (<10 menit).
> - BGE-M3 hanya perlu embed **queries** (bukan corpus — corpus sudah diindex di NB03): 1.190 × 2 = 2.380 embeddings, selesai dalam hitungan menit.
> - Load `reranker_precision_table.json` dari Kaggle Dataset `crosslingual-rag-results` (output NB04):
>
>   ```python
>   import json
>   with open("/kaggle/input/crosslingual-rag-results/reranker_precision_table.json") as f:
>       reranker_table = json.load(f)
>   ```
>
> - Input notebook ini: attach `crosslingual-rag-data`, `crosslingual-rag-indexes`, dan `crosslingual-rag-results`.

---

### `08_evaluation_ragas.ipynb`

**Tujuan:** RAGAS evaluation untuk **Full System saja** (Phase 4). HotpotQA tidak dipakai di phase ini. Output notebook ini mengisi **paper Table VI (RAGAS Full System)**.

**Corpus: Wikipedia ID (~20.000 artikel)** — sama dengan NB07. Full System dijalankan fresh terhadap Wikipedia ID — bukan reusing ablation outputs dari NB06 (yang menggunakan dataset-native corpus).

**Dua LLM yang dievaluasi (keduanya wajib):**

- **Qwen3-8B** (local, 4-bit quantized) — untuk kontinuitas dengan ablation study
- **Gemini-2.5-Flash** (API) — untuk perbandingan reasoning model vs non-reasoning

**Dataset:**

- **TyDiQA-ID eval** (1.452 instances) — ID queries only
- **XQuAD-ID eval** (1.190 instances) — dijalankan dua kali: EN queries dan ID queries

**Sample cap:** ≤500 samples per dataset per LLM. Pilih secara stratified/random, log seed untuk reproducibility.

**RAGAS metrics:** Faithfulness, Factual Correctness, Semantic Similarity, RAGAS Avg (unweighted mean).

**Catatan TyDiQA-ID:** Tidak ada EN query counterpart — semua TyDiQA-ID results reflect ID-query performance only. Tidak perlu dijalankan dua kali untuk dataset ini.

**Target comparison:** Setiawan et al. [4] Gemini-2.5-Flash avg 0.8207 pada institutional corpus (30 dokumen). Perbandingan bersifat kontekstual — corpus size dan domain berbeda jauh. Row Setiawan et al. hanya untuk kontekstualisasi score range, bukan controlled benchmark comparison.

**Paper Table VI yang diisi notebook ini:**

```
| System              | LLM              | Dataset       | Faithfulness | Factual Correctness | Semantic Similarity | RAGAS Avg |
| Setiawan et al. [4] | —                | Institutional | 0.9195       | 0.6274              | 0.9151              | 0.8207    |  ← hardcoded, referensi eksternal
| Full System         | Qwen3-8B         | TyDiQA-ID     | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
| Full System         | Qwen3-8B         | XQuAD-ID (ID) | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
| Full System         | Qwen3-8B         | XQuAD-ID (EN) | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
| Full System         | Gemini-2.5-Flash | TyDiQA-ID     | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
| Full System         | Gemini-2.5-Flash | XQuAD-ID (ID) | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
| Full System         | Gemini-2.5-Flash | XQuAD-ID (EN) | [NB8]        | [NB8]               | [NB8]               | [NB8]     |
```

**Output files:**

```
results/phase4/ragas_qwen_tydiqa.json
results/phase4/ragas_qwen_xquad_id.json
results/phase4/ragas_qwen_xquad_en.json
results/phase4/ragas_gemini_tydiqa.json
results/phase4/ragas_gemini_xquad_id.json
results/phase4/ragas_gemini_xquad_en.json
results/phase4/ragas_summary_table.csv
```

> 🟡 **Kaggle T4 Note — NB08:**
>
> - Install RAGAS dan dependencies di awal session:
>
>   ```python
>   !pip install -q ragas langchain-google-genai
>   ```
>
> - Gunakan **Gemini-2.5-Flash sebagai RAGAS judge** untuk kedua LLM (Qwen dan Gemini) — lebih efisien daripada load Qwen sebagai judge (hemat VRAM):
>
>   ```python
>   from ragas import evaluate
>   from ragas.llms import LangchainLLMWrapper
>   from langchain_google_genai import ChatGoogleGenerativeAI
>   import os
>   os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
>   judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash"))
>   ```
>
> - **Sample cap 500 adalah wajib** — untuk manajemen API cost Gemini dan GPU time. Set `random_seed=42` dan log seed di output file.
> - Unzip FAISS Wikipedia ID index di awal session (sama seperti NB07).
> - **Simpan setiap `ragas_*.json` segera setelah selesai** — jangan tunggu semua 6 kombinasi beres. Upload ke Kaggle Dataset `crosslingual-rag-results` per file.
> - RAGAS evaluation untuk 500 samples estimasi ~2–3 jam (mayoritas waktu adalah API call latency Gemini). Gunakan `tqdm` agar sesi tidak timeout.
> - Row Setiawan et al. (0.9195 / 0.6274 / 0.9151) adalah **hardcoded** di summary table — bukan hasil notebook ini.
> - Input notebook ini: attach `crosslingual-rag-data`, `crosslingual-rag-indexes`, dan model checkpoint dari HF Hub.

---

## Dependency Graph

```
01_data_prep
    ├── 02_complexity_classifier
    ├── 03_retrieval_indexing
    │       └── 04_reranker
    │               └── 05_generation_pipeline (Phase 1)
    │                       └── 06_ablation_study (Phase 2)
    │                               ├── 07_dgap_measurement (Phase 3)
    │                               └── 08_evaluation_ragas (Phase 4)
    └── (data shapes confirmed before any indexing)
```

Notebooks 01–04 dapat dijalankan paralel setelah `01` selesai. Notebooks 05–08 harus berurutan.

> 🟡 **Kaggle T4 Note — Artifact Handoff Antar Session:**
>
> Setiap Kaggle session adalah ephemeral. Gunakan tiga Kaggle Datasets sebagai persistent layer:
>
> | Kaggle Dataset             | Dibuat oleh |   Dipakai oleh   | Isi utama                                   |
> | -------------------------- | :---------: | :--------------: | ------------------------------------------- |
> | `crosslingual-rag-data`    |    NB01     |    NB02–NB08     | Semua JSONL splits + Wikipedia corpus       |
> | `crosslingual-rag-indexes` |    NB03     | NB04, NB06–NB08  | BM25 pickles + FAISS indexes                |
> | `crosslingual-rag-results` |  NB02–NB08  | NB06–NB08, paper | JSON/CSV hasil evaluasi, model eval metrics |
>
> Model checkpoint NB02 di-push ke **HuggingFace Hub** (`username/mdeberta-complexity-id`, private) dan di-load dari sana di notebook manapun yang butuh classifier.

---

## Implementation Notes

- **No translation anywhere** — seluruh pipeline translation-free. BGE-M3 handles cross-lingual natively.
- **Loss tidak dilaporkan** sebagai evaluation metric di notebook manapun. Loss hanya muncul di training loop NB02 sebagai monitor, bukan sebagai output tabel.
- **Dua corpus, dua tujuan berbeda:**
  - HotpotQA source docs → ablation only (NB06), dataset-native
  - Wikipedia ID (~20K artikel) → Δgap (NB07) dan RAGAS (NB08), open-domain
  - `id_newspapers_2018` **tidak digunakan** di pipeline ini
- **XQuAD-ID parallel structure** harus dijaga dari NB01 — jangan split `question_en` dan `question_id` ke file terpisah. Pertahankan `question_id` yang sama sebagai key untuk Δgap calculation di NB07.
- **TyDiQA-ID** tidak masuk ablation (NB06) — hanya untuk classifier training (NB02) dan eval di Δgap + RAGAS (NB07, NB08).
- **Accuracy tidak dilaporkan** di manapun sebagai evaluation metric — hanya EM dan F1 untuk generation, Recall@10/MRR/P@1/P@3 untuk retrieval, RAGAS untuk grounding.
- **mDeBERTa-v3-base** adalah classifier yang digunakan — bukan IndoBERT. Model string: `microsoft/mdeberta-v3-base`.

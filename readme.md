# Cross-lingual RAG for Indonesian–English QA

Note: the notebook is be run on the kaggle, because it have virtual GPU, for the better researching

## External Link (Kaggle)

### Notebook

1. NB01 - Data Preparation - https://www.kaggle.com/code/melisaolivia/01-data-prep

### Dataset

1. crosslingual-rag-data - https://www.kaggle.com/datasets/melisaolivia/crosslingual-rag-data

## Research Phase Track

| #   | Phase                                 | Dataset                                                             | Proses Singkat                                                                                                                                                                                                  | Arsitektur                                                            | Metrik                                                                    |
| --- | ------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 0   | **Data Preparation ✅**               | HotpotQA, XQuAD-ID, TyDiQA-ID                                       | Download, filter missing pairs; HotpotQA: ~10.000 train + 7.405 eval (label "C" hardcoded); TyDiQA-ID: split stratified 80/20 → 4.815 train + 1.452 eval; XQuAD-ID: 1.190 full untuk eval saja (tidak di-split) | —                                                                     | —                                                                         |
| 1   | **Complexity Classifier Fine-tuning** | Train: HotpotQA (~10.000, class C) + TyDiQA-ID (4.815, class A/B/C) | Fine-tune mDeBERTa-v3-base dengan weighted loss (kompensasi class imbalance karena HotpotQA dominan di class C); XQuAD-ID tidak dipakai untuk training                                                          | mDeBERTa-v3-base                                                      | Accuracy, Precision, Recall, F1, Confusion Matrix                         |
| 2   | **Baseline Reproduction**             | HotpotQA eval only                                                  | Replikasi hasil baseline pada HotpotQA — kasus kegagalan paling ekstrem (multi-retrieval collapse 0.007); tidak memerlukan classifier karena semua pertanyaan langsung dilabeli "C"                             | BM25 Index; no classifier for HotpotQA                                | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                    |
| 3   | **Ablation Study**                    | HotpotQA eval, XQuAD-ID eval (ID query & EN query dipisah)          | Evaluasi progresif per config; tiap config dijalankan pada dataset-native corpus                                                                                                                                | BM25 → BGE-M3 Dense → BGE-M3 Sparse+Dense → +mMiniLM Reranker → +CRAG | HotpotQA: EM, F1, Mean Steps; XQuAD-ID EN & ID: EM, F1                    |
| 4   | **Δgap Measurement**                  | XQuAD-ID eval, TyDiQA-ID eval; corpus: Wikipedia ID                 | Query ID dan EN identik dijalankan pada Wikipedia ID corpus; hitung selisih per retrieval config                                                                                                                | BGE-M3 Sparse+Dense + mMiniLM Reranker                                | Recall@10, MRR, Δgap per config (NDCG@10 dropped — binary relevance only) |
| 5   | **RAGAS Evaluation**                  | XQuAD-ID eval, TyDiQA-ID eval; corpus: Wikipedia ID                 | Evaluasi grounding kualitas jawaban full-system                                                                                                                                                                 | Full System (BGE-M3 + Reranker + CRAG + Reasoning LLM)                | Faithfulness, Factual Correctness, Semantic Similarity                    |

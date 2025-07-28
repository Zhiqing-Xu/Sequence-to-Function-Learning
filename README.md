# Protein Sequence-to-Function Learning with Pretrained Language Models

This repository presents a benchmarking framework for sequence-to-function prediction using frozen protein language model (pLM) embeddings. A novel pooling strategy—convolutional pooling—is proposed to extract task-specific features from fixed pLM representations. The approach is evaluated across four benchmark datasets and compared to strong baselines, demonstrating competitive performance while maintaining computational efficiency.

## Pipeline Overview

The figure below illustrates the full workflow for protein sequence-to-function learning using pLM embeddings. Protein sequences are first embedded using pretrained models (e.g., ESM), followed by task-specific feature aggregation using either self-attention pooling or convolutional pooling. The resulting representations are used to predict functional properties such as activity or fitness.

<p align="center">
  <img src="_Figures/_PhD_Figure_SeqToFunc_Intro_pLMbased.png" alt="Sequence-to-Function Pipeline" width="600">
</p>

---

## Benchmarking Results

### 1. GB1 Fitness Prediction (FLIP Benchmark)

| Model                     | 1-vs-rest | 2-vs-rest | 3-vs-rest | low-vs-high |
|--------------------------|-----------|-----------|-----------|--------------|
| ESM-1b + ConvPool        | 0.310 ± 0.020 | **0.646 ± 0.006** | **0.878 ± 0.003** | 0.433 ± 0.010 |
| ESM-1b + AttPool         | **0.334 ± 0.012** | 0.619 ± 0.007 | 0.852 ± 0.005 | **0.447 ± 0.004** |
| FLIP (Best Baseline)     | 0.32      | 0.59      | 0.83      | 0.59         |

Both pooling strategies achieve strong results across the FLIP GB1 splits, with convolutional pooling showing superior performance in high-data regimes.

---

### 2. β-Lactamase Activity Prediction

<p align="center">
  <img src="_Figures/_PhD_Figure_SeqsToFunc_Results_BetaLactamase.png" alt="Beta-Lactamase Results" width="600">
</p>

| Model                    | Parameters  | Spearman’s ρ       |
|-------------------------|-------------|---------------------|
| ESM-1b + ConvPool       | ~1.7M       | **0.886 ± 0.007**   |
| ESM-1b + AttPool        | ~1.4M       | 0.828 ± 0.014       |
| Fine-Tuned ESM-1b       | ~650M       | 0.839 ± 0.053       |
| ESM-1b + AvgPool        | ~0.9M       | 0.528 ± 0.009       |

Convolutional pooling outperforms all baselines, including full model fine-tuning, while requiring far fewer parameters.

---

### 3. PafA Enzymatic Activity Prediction

<p align="center">
  <img src="_Figures/_PhD_Figure_SeqsToFunc_Results_PafA.png" alt="PafA Results" width="600">
</p>

| Pooling Method      | Model     | Pearson’s r        | Spearman’s ρ       |
|---------------------|-----------|---------------------|---------------------|
| Convolutional Pool  | ESM-2     | 0.594 ± 0.072       | 0.600 ± 0.059       |
| Self-Attention Pool | ESM-2     | **0.619 ± 0.047**   | **0.641 ± 0.039**   |

Self-attention pooling performs best on this challenging mutation dataset, where the task is to predict the effect of substitutions not observed during training.

---

### 4. avGFP Fluorescence Landscape Prediction

| Model                       | Parameters       | Spearman’s ρ         |
|----------------------------|------------------|-----------------------|
| ESM-1b + ConvPool          | ~5.5M            | 0.662 ± 0.002         |
| ESM-1b + AttPool           | ~3.5M            | 0.651 ± 0.003         |
| Fine-Tuned ESM-1b          | ~650M            | **0.679 ± 0.002**     |
| ESM-1b + AvgPool           | ~0.9M            | 0.430 ± 0.002         |
| eUniRep + Ridge Regression | Non-neural model | 0.427 ± 0.003         |

The proposed pooling methods significantly outperform baseline approaches with minimal training overhead.

---

## Summary

This framework demonstrates that simple yet effective pooling strategies applied to pretrained pLM embeddings can achieve competitive performance across diverse protein function prediction tasks. Convolutional pooling, in particular, offers strong predictive power for mutation effect datasets with high sequence similarity, while attention pooling excels in tasks requiring generalization to unseen residues.



### Pipeline.

- N00_Data_Preprocessing.py     : Preprocess and the dataset.
- N03_LM_Embeddings.py          : Get sequence embeddings.
- N05X_SQemb_y.py               : Train the model and evaluate.






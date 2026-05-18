# Torch-RecHub Recommender Systems Practice

<div align="center">
  <a href="https://pypi.org/project/torch-rechub/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-rechub?style=for-the-badge&color=005A9C&label=torch-rechub"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-012A4A?style=for-the-badge"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-1.10%2B-EE4C2C?style=for-the-badge"></a>
  <a href="https://rocm.docs.amd.com/"><img alt="ROCm" src="https://img.shields.io/badge/AMD%20ROCm-ready-ED1C24?style=for-the-badge"></a>
  <a href="https://github.com/datawhalechina/torch-rechub"><img alt="GitHub stars" src="https://img.shields.io/github/stars/datawhalechina/torch-rechub?style=for-the-badge&color=00C8FF"></a>
</div>

This directory introduces practical recommender system workflows with [Torch-RecHub](https://github.com/datawhalechina/torch-rechub) on AMD GPU / ROCm environments. The goal is not only to run several models, but to understand how a modern recommendation pipeline is organized: feature definitions, data generators, trainers, ranking and matching models, experiment tracking, and model export for serving.

Torch-RecHub is a lightweight PyTorch recommender system framework designed to make industrial-style recommendation experiments easier to reproduce. It provides 30+ mainstream models across ranking, matching, multi-task learning, and generative recommendation, together with unified data loading, training, evaluation, tracking, and ONNX export utilities.

## Why It Is Here

In hello-rocm, this case focuses on the AMD GPU side of recommender systems. You will learn how to install ROCm-compatible PyTorch, run Torch-RecHub tutorials on AMD hardware, and connect model training with deployment-oriented steps such as ONNX export and lightweight retrieval validation.

The tutorials use small sample datasets so the full workflow can be tested locally. The metrics are for pipeline validation rather than production benchmarking.

## What You Will Learn

- How CTR ranking models such as DeepFM turn dense and sparse features into click-through-rate predictions.
- How DIN models user behavior sequences with target-aware attention.
- How DSSM builds a two-tower matching model and uses item embeddings for retrieval.
- How MMOE shares experts across multiple recommendation objectives.
- How Torch-RecHub connects WandB, SwanLab, and TensorBoardX through a unified `model_logger` interface.
- How trained ranking and matching models can be exported to ONNX and validated with ONNXRuntime.

## Torch-RecHub Highlights

- **PyTorch-first design**: uses PyTorch dynamic graphs and hardware acceleration, including CPU, NVIDIA CUDA GPU, AMD ROCm GPU, and Huawei Ascend NPU.
- **Rich model library**: includes 30+ recommendation models covering ranking, matching, multi-task learning, and generative recommendation.
- **Modular components**: feature definitions, layers, trainers, losses, metrics, and utilities are separated so models and datasets can be extended.
- **Standardized workflow**: provides unified data loading, training, evaluation, experiment tracking, and model export paths.
- **Deployment awareness**: supports ONNX export and serving-oriented validation, making it easier to move from training to inference.
- **Experiment tracking**: integrates WandB, SwanLab, and TensorBoardX through a consistent logging interface.

## Tutorials

1. [CTR Prediction: DeepFM](./00_QuickStart_CTR_DeepFM.md)
2. [Sequential Interest Modeling: DIN](./01_Ranking_DIN.md)
3. [Matching / Retrieval: DSSM](./02_Matching_DSSM.md)
4. [Multi-task Learning: MMOE](./03_MultiTask_MMOE.md)
5. [Experiment Tracking: model_logger](./04_Experiment_Tracking_Light.md)
6. [Model Export and Inference Validation: ONNX](./05_Model_Export_and_Serving.md)

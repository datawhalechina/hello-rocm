# Torch-RecHub 推荐系统实战

<p align="center">
  <a href="https://pypi.org/project/torch-rechub/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-rechub?style=for-the-badge&color=005A9C&label=torch-rechub"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-012A4A?style=for-the-badge"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-1.10%2B-EE4C2C?style=for-the-badge"></a>
  <a href="https://rocm.docs.amd.com/"><img alt="ROCm" src="https://img.shields.io/badge/AMD%20ROCm-ready-ED1C24?style=for-the-badge"></a>
  <a href="https://github.com/datawhalechina/torch-rechub"><img alt="GitHub stars" src="https://img.shields.io/github/stars/datawhalechina/torch-rechub?style=for-the-badge&color=00C8FF"></a>
</p>

本目录收录 [Torch-RecHub](https://github.com/datawhalechina/torch-rechub) 在 AMD GPU / ROCm 环境下的推荐系统实战教程。目标不只是跑通几个模型，而是理解一个现代推荐系统训练链路如何组织：特征定义、数据生成器、训练器、排序与召回模型、实验跟踪，以及面向服务部署的模型导出。

Torch-RecHub 是一个轻量、高效、易用的 PyTorch 推荐系统框架，面向工业级推荐实验的快速复现与扩展。项目提供 30+ 主流推荐模型，覆盖精排、召回、多任务学习和生成式推荐，并提供统一的数据加载、训练评估、实验跟踪和 ONNX 导出能力。

## 为什么放在 hello-rocm

在 hello-rocm 中，这组案例聚焦 **AMD GPU / ROCm 上的推荐系统实践**。你会学习如何安装 ROCm 适配的 PyTorch，如何在 AMD 硬件上运行 Torch-RecHub 教程，并把训练流程进一步连接到 ONNX 导出、轻量召回验证等部署相关环节。

这些教程使用轻量样例数据，适合在本地快速验证完整流程。文中的指标主要用于确认链路跑通，不代表生产数据集上的最终效果。

## 你会学到什么

- DeepFM 如何把 dense / sparse 特征用于 CTR 预估。
- DIN 如何用候选物品对用户历史行为序列做 attention。
- DSSM 如何构建双塔召回模型，并使用 item embedding 做检索。
- MMOE 如何在多个推荐目标之间共享 expert 并分配任务 tower。
- Torch-RecHub 如何通过统一的 `model_logger` 接入 WandB、SwanLab、TensorBoardX。
- 排序与召回模型如何导出 ONNX，并用 ONNXRuntime 做最小推理验证。

## Torch-RecHub 项目特色

- **PyTorch 优先**：基于 PyTorch 动态图和硬件加速能力，支持 CPU、NVIDIA CUDA GPU、AMD ROCm GPU 和华为昇腾 NPU。
- **模型库丰富**：内置 30+ 推荐模型，覆盖 ranking、matching、multi-task learning 和 generative recommendation。
- **模块化设计**：特征、层、训练器、损失、指标和工具相互解耦，方便扩展新模型和新数据集。
- **标准化流程**：提供统一的数据加载、训练、评估、实验跟踪和模型导出路径。
- **部署友好**：支持 ONNX 导出和推理验证，帮助从训练走向服务化。
- **实验跟踪**：统一接入 WandB、SwanLab、TensorBoardX，便于记录超参和指标。

## 教程目录

1. [CTR 预测：DeepFM](/zh/05-amd-yes/torch-rechub/00_QuickStart_CTR_DeepFM)
2. [序列兴趣建模：DIN](/zh/05-amd-yes/torch-rechub/01_Ranking_DIN)
3. [匹配/召回：DSSM](/zh/05-amd-yes/torch-rechub/02_Matching_DSSM)
4. [多任务学习：MMOE](/zh/05-amd-yes/torch-rechub/03_MultiTask_MMOE)
5. [实验跟踪：model_logger 接入](/zh/05-amd-yes/torch-rechub/04_Experiment_Tracking_Light)
6. [模型导出与推理验证：ONNX](/zh/05-amd-yes/torch-rechub/05_Model_Export_and_Serving)

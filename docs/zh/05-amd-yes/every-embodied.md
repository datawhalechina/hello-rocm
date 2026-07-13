# Every Embodied：在 ROCm 上复刻具身智能策略

<div align="center">

[![AMD](https://img.shields.io/badge/AMD-ROCm-ED1C24)](https://rocm.docs.amd.com/)
[![Every Embodied](https://img.shields.io/badge/Datawhale-Every%20Embodied-1F6FEB)](https://github.com/datawhalechina/every-embodied)

</div>

ROCm 不只可以部署和微调语言模型，也可以承担具身智能中的视觉编码、策略训练、动作推理与批量评估。Datawhale 的 [Every Embodied](https://github.com/datawhalechina/every-embodied) 项目整理了一套 MuJoCo 桌面抓取实验，并在 AMD Ryzen AI MAX+ 395 / ROCm 环境中复刻 ACT、SmolVLA、Pi0 与 Pi0.5 的训练和调试流程。

本文只介绍项目入口和学习路线。完整代码、Notebook、环境命令、实验结果与排障过程均在 Every Embodied 持续维护，避免同一套教程在两个仓库重复更新。

<div align="center">
  <img src="../../public/images/05-amd-yes/every-embodied/pnp_four_view_strict_success_sequence.jpg" alt="MuJoCo 四视角抓杯严格成功序列" width="95%" />
  <p>MuJoCo 四视角抓杯严格成功序列</p>
</div>

## 能学到什么

这套专题不是只展示一段成功视频，而是覆盖从数据到物理成功判定的完整闭环：

1. 在 MuJoCo 中进行键盘遥操作，录制多视角图像、机器人状态和动作轨迹。
2. 将数据整理为 LeRobot 数据格式，并检查 episode、时间戳、动作维度与归一化统计。
3. 在 ROCm 上训练 ACT、SmolVLA、Pi0/Pi0.5，并记录显存、温度、loss 和 checkpoint。
4. 把策略部署回仿真环境做闭环推理，批量评估随机位置和不同颜色的杯子。
5. 使用“抬起、搬运、释放、最终落点”组成的严格物理成功条件，而不是只看 loss 或单帧画面。
6. 通过 teacher-forced 回放、open-loop action trace、DAgger/recovery 数据、加权采样和动作表示审计定位失败原因。

## 模型与实验内容

| 模型 | 专题中的主要内容 | 当前阅读重点 | Notebook |
|:---|:---|:---|:---|
| ACT | 单条与随机化数据训练、关节动作诊断、DAgger/recovery 数据 | 小数据模仿学习为什么会在闭环中累积误差 | [ACT ROCm 训练](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/08_act_training_rocm.ipynb) |
| SmolVLA | 预训练权重加载、红蓝杯采样失衡、加权采样与随机位置评估 | 数据覆盖和采样分布如何影响泛化 | [SmolVLA ROCm 训练](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/09_smolvla_training_rocm.ipynb) |
| Pi0 | 权限与权重 smoke test、raw policy 与学习辅助头分离评估 | 不把 scripted finisher 的成功误报为端到端 VLA 成功 | [Pi0 ROCm 训练](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/10_pi0_training_rocm.ipynb) · [严格端到端诊断](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/12_pi0_strict_input_end_to_end.ipynb) |
| Pi0.5 | LeRobot v3 数据规范、EEF-delta 动作表示、chunk 对齐和 recovery 训练 | 动作方向、执行窗口和闭环纠偏为什么比单纯增加训练步数更关键 | [Pi0.5 随机位置与 EEF-delta](https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/13_pi05_random_position_eef_delta.ipynb) |

专题会明确区分三种结果口径：**raw policy**、只使用视觉/语言/本体状态的**可学习辅助模块**，以及注入目标位置或手写阶段规则的**诊断脚手架**。三者用途不同，成功率不能混在一起比较。

## 学习入口

- [Every Embodied 项目 README](https://github.com/datawhalechina/every-embodied#readme)
- [AMD ROCm 策略复刻专题（GitHub）](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98)
- [Every Embodied 在线阅读](https://datawhalechina.github.io/every-embodied/zh-cn/#/doc/MTYt5LiT6aKY57uE6Zif5a2m5LmgLzA0LUFNRC1ST0Nt562W55Wl5aSN5Yi75LiT6aKYL1JFQURNRS5tZA)
- [AMD Radeon Cloud 使用指南](/zh/cloud/amd-radeon-cloud)

建议按专题编号顺序学习。首次接触项目时，先运行设备检查和物理成功评估 Notebook，再进入数据采集、训练和部署；这样可以先确认“评估器真的能判对”，避免训练完成后才发现动作方向、数据格式或成功条件有误。

## 硬件与软件说明

专题的主要验证设备为 AMD Ryzen AI MAX+ 395，使用 Ubuntu、ROCm、PyTorch、LeRobot 与 MuJoCo。Ryzen AI MAX+ 395 采用统一内存架构，系统内存与 GPU 可用内存共享，因此训练时仍需同时观察 GPU 分配、系统内存和温度，不能把全部统一内存都视为可无条件使用的独立显存。

不同 ROCm、PyTorch 和 LeRobot 版本的兼容性会变化。开始复现前，请以专题中的设备检查 Notebook、锁定版本和当前 [ROCm 兼容性文档](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) 为准。

## 为什么值得尝试

具身智能策略迁移到 ROCm 后，真正困难的部分通常不是把 `cuda` 字符串改成 `hip`，而是把模型依赖、数据格式、动作语义、执行频率和物理评估协议同时对齐。Every Embodied 把这些失败案例和修复工具一并保留下来，可以直接看到一个策略从“loss 正常”到“仿真中真正完成任务”之间还缺哪些工程环节。

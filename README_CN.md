[English](README.md) | 简体中文

![](https://release-data.cdn.bcebos.com/Quanlse_title_cn.png)

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) [![](https://img.shields.io/badge/build-passing-green)]() ![](https://img.shields.io/badge/Python-3.6--3.8-blue) ![](https://img.shields.io/badge/release-v1.0.0-blue)

[量脉(Quanlse)](https://quanlse.baidu.com)是由百度研究院[量子计算研究所](https://quantum.baidu.com)开发的基于云服务的量子控制平台。量脉的目标是搭起连接量子软件和硬件之间的桥梁。通过强大的量脉云服务和开源 SDK，量脉提供高效和专业的量子控制解决方案。

量脉支持任意单量子比特门脉冲和双量子比特门脉冲的产生和调度。借助量脉的工具箱，用户还可以实现物理设备的建模、动力学演化的模拟和错误分析的可视化。更有趣的是，用户可以使用量脉的工具在脉冲层面实现量子算法。此外，量脉还支持量子控制领域的高级研发工作。

[TOC]

## 安装

我们强烈建议使用 [Anaconda](https://www.anaconda.com/) 作为研发环境，以提供最佳体验。

### 通过 pip 安装

我们推荐通过 `pip` 完成安装，

```bash
pip install Quanlse
```

### 通过 Github 下载安装

用户也可以选择下载全部文件后进行本地安装，

```bash
git clone http://github.com/baidu/Quanlse
cd Quanlse
pip install -e .
```

### 运行示例程序

现在，您可以尝试运行示例程序来验证量脉是否已成功安装，

```bash
cd Example
python 1-Example-Pi-Pulse.py
```

## 入门和开发

### 概述
在开始使用量脉之前，我们建议用户首先通过阅读[简介](https://quanlse.baidu.com/#/doc/overview)了解该平台的概览。然后，[快速入门](https://quanlse.baidu.com/#/doc/quickstart)将会一步一步地指导您如何使用量脉云服务，以及如何使用量脉来构建您的第一个程序。接下来，我们鼓励用户在[教程](https://quanlse.baidu.com/#/doc/tutorial-overview)里学习更多量脉提供的功能和应用程序。最后，我们鼓励用户能够使用量脉解决问题。有关量脉 API 的完整和详细文档，请阅读我们的 [API](https://quanlse.baidu.com/api/) 文档页.

### 教程

量脉提供了从基础到进阶主题的详尽教程。目前每个教程都支持用户在我们的[网站](https://quanlse.baidu.com/#/doc/tutorial-overview)上阅读。对于有兴趣的开发人员，我们建议下载并且使用 [Jupyter Notebooks](https://github.com/baidu/Quanlse/tree/master/Tutorial)。教程的内容如下：

- **单比特量子门**
  1. [单比特量子门](https://quanlse.baidu.com/#/doc/tutorial-single-qubit)
  2. [校准 Pi 脉冲](https://quanlse.baidu.com/#/doc/tutorial-pi-pulse)
- **两比特量子门**
  1. [iSWAP 门](https://quanlse.baidu.com/#/doc/tutorial-iswap)
  2. [Controlled-Z 门](https://quanlse.baidu.com/#/doc/tutorial-cz)
  3. [Cross-Resonance 门](https://quanlse.baidu.com/#/doc/tutorial-cr)
- **进阶应用**
  1. [DRAG 脉冲](https://quanlse.baidu.com/#/doc/tutorial-drag)
  2. [误差分析](https://quanlse.baidu.com/#/doc/tutorial-error-analysis)
  3. [用于量子电路的量脉调度器](https://quanlse.baidu.com/#/doc/tutorial-scheduler)
  4. [基于脉冲的变分量子本征求解算法](https://quanlse.baidu.com/#/doc/tutorial-pbvqe)

此外，量脉还支持用于核磁共振（NMR）量子计算的量子控制服务。关于[量脉核磁](https://nmr.baidu.com/)的教程，请阅读[教程：量脉核磁](https://quanlse.baidu.com/#/doc/nmr)。

## 反馈

我们鼓励用户通过 [Github Issues](https://github.com/baidu/Quanlse/issues) 或 quanlse@baidu.com 联系我们反馈一般问题、错误和改进意见。我们希望通过与社区的合作让量脉变得更好！

## 常见问题
**Q：我应该如何开始使用量脉？**

**A：**我们建议用户访问我们的[网站](https://quanlse.baidu.com/#/doc/quickstart)并遵循以下路线图：

**步骤1：**进入[快速入门](https://quanlse.baidu.com/#/doc/quickstart)了解如何访问量脉云服务。

**步骤2：**学习[单量子比特控制](https://quanlse.baidu.com/#/doc/tutorial-single-qubit)和[双量子比特控制](https://quanlse.baidu.com/#/doc/tutorial-iswap)的例子来熟悉量脉。

**步骤3：**研究更多[进阶应用](https://quanlse.baidu.com/#/doc/tutorial-drag)，探索量脉更多的可能性。

**Q：我的 credit points 用完了该怎么办？**

**A： **请通过 [Quantum Hub](https://quantum-hub.baidu.com) 联系我们。首先，登录 [Quantum Hub](https://quantum-hub.baidu.com)，然后进入“用户管理->反馈建议”页面，并输入必要的信息（选择“获取更多点数”）。提交您的反馈并等待回复。

**Q：我的应该如何在研究工作中引用量脉？**

**A：**我们鼓励研发人员使用量脉进行量子控制领域的相关工作，请通过如下 [BibTeX 文件](Quanlse.bib)引用量脉。
## 更新日志

量脉的更新日志可在 [CHANGELOG.md](CHANGELOG.md) 文件中查看。

## 版权和许可证

量脉使用 [Apache-2.0 license](LICENSE) 作为许可证。

## 参考文献

[1] [Quantum Computing - Wikipedia](https://en.wikipedia.org/wiki/Quantum_computing).

[2] Nielsen, Michael A., and Isaac L. Chuang. *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge: Cambridge UP, 2010. Print.

[3] [Werschnik, J., and E. K. U. Gross. "Quantum optimal control theory." *Journal of Physics B: Atomic, Molecular and Optical Physics* 40.18 (2007): R175.](https://doi.org/10.1088/0953-4075/40/18/R01)

[4] [Wendin, Göran. "Quantum information processing with superconducting circuits: a review." *Reports on Progress in Physics* 80.10 (2017): 106001.](https://doi.org/10.1088/1361-6633/aa7e1a)

[5] [Krantz, Philip, et al. "A quantum engineer's guide to superconducting qubits." *Applied Physics Reviews* 6.2 (2019): 021318.](https://doi.org/10.1063/1.5089550)
English | [简体中文](README_CN.md)

![](https://release-data.cdn.bcebos.com/Quanlse_title_en.png)

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) [![](https://img.shields.io/badge/build-passing-green)]() ![](https://img.shields.io/badge/Python-3.6--3.8-blue) ![](https://img.shields.io/badge/release-v1.0.0-blue)

[Quanlse (量脉)](https://quanlse.baidu.com) is a cloud-based platform for quantum control developed by the [Institute for Quantum Computing](https://quantum.baidu.com) at Baidu Research. Quanlse aims to bridge the gap between quantum software and hardware. It provides efficient and professional quantum control solutions via an open-source SDK strengthened by Quanlse Cloud Service.

Quanlse supports the pulse generation and scheduling of arbitrary single-qubit and two-qubit gates. With the help of toolkits in Quanlse, one can also use Quanlse for modeling physical devices, simulating dynamical evolution, and visualizing error analysis. More interestingly, users can achieve pulse-level control of quantum algorithms using Quanlse. Furthermore, Quanlse also supports advanced research and development (R&D) in the field of quantum control.

## Install

We strongly recommend using [Anaconda](https://www.anaconda.com/) for your R&D environment and upgrading the requirements to the latest versions for the best experience.

### Install via pip

We recommend the following way of installing Quanlse with `pip`,

```bash
pip install Quanlse
```

### Download and install via GitHub

You can also download all the files and install Quanlse locally,

```bash
git clone http://github.com/baidu/Quanlse
cd Quanlse
pip install -e .
```

### Run programs

Now, you can try to run a program to verify whether Quanlse has been installed successfully.

```bash
cd Example
python 1-Example-Pi-Pulse.py
```

## Introduction and developments

### Overview

To get started with Quanlse, users are recommended to go through the [Overview](https://quanlse.baidu.com/#/doc/overview) firstly to acquire the whole picture of this platform. [Quick Start](https://quanlse.baidu.com/#/doc/quickstart) could then be a good place to guide you on how to use Quanlse Cloud Service step by step and how to construct your first program using Quanlse. Next, users are encouraged to learn more functions and applications from the [tutorials](https://quanlse.baidu.com/#/doc/tutorial-overview) Quanlse provided. Finally, it would be great if users could solve their own problems using Quanlse. For complete and detailed documentation of the Quanlse API, please refer to our [API documentation](https://quanlse.baidu.com/api/).

### Tutorials

Quanlse provides detailed and comprehensive tutorials from fundamental to advanced topics. Each tutorial currently supports reading on our [website](https://quanlse.baidu.com). For interested developers, we recommend them to download [Jupyter Notebooks](https://github.com/baidu/Quanlse/tree/master/Tutorial/EN) and play with it. The tutorial list is as follows:

- **Single-Qubit Control**
    - [Single-Qubit Gate](https://quanlse.baidu.com/#/doc/tutorial-single-qubit)
    - [Calibrate Pi Pulses](https://quanlse.baidu.com/#/doc/tutorial-pi-pulse)
- **Two-Qubit Control**
    - [iSWAP Gate](https://quanlse.baidu.com/#/doc/tutorial-iswap)
    - [Controlled-Z Gate](https://quanlse.baidu.com/#/doc/tutorial-cz)
    - [Cross-Resonance Gate](https://quanlse.baidu.com/#/doc/tutorial-cr)
- **Advanced Applications**
    - [Derivative Removal by Adiabatic Gate](https://quanlse.baidu.com/#/doc/tutorial-drag)
    - [Error Analysis](https://quanlse.baidu.com/#/doc/tutorial-error-analysis)
    - [Quanlse Scheduler](https://quanlse.baidu.com/#/doc/tutorial-scheduler)
    - [Pulse-based Variational Quantum Eigensolver Algorithm](https://quanlse.baidu.com/#/doc/tutorial-pbvqe)

In addition, Quanlse also supports quantum control for nuclear magnetic resonance (NMR) quantum computing. For more information about [QuanlseNMR](https://nmr.baidu.com/en/), please refer to the [tutorial: QuanlseNMR](https://quanlse.baidu.com/#/doc/nmr).


## Feedbacks

Users are encouraged to contact us through [Github Issues](https://github.com/baidu/Quanlse/issues) or quanlse@baidu.com with general questions, bugs, and potential improvements. We hope to make Quanlse better together with the community!

## Frequently Asked Questions

**Q: How should I get started with Quanlse?**

**A:** We recommend users go to our [website](http://quanlse.baidu.com) and follow the roadmap. 

- **Step 1:** Go to [Quick Start](https://quanlse.baidu.com/#/doc/quickstart) to learn how to access Quanlse Cloud Service.
- **Step 2:** Get familiarized with Quanlse by going through the examples of [single-qubit control](https://quanlse.baidu.com/#/doc/tutorial-single-qubit) and [two-qubit control](https://quanlse.baidu.com/#/doc/tutorial-iswap). 
- **Step 3:** Explore more possibilities with Quanlse by studying more [advanced applications](https://quanlse.baidu.com/#/doc/tutorial-drag).

**Q: What should I do when I run out of my credit points?**  

**A:** Please contact us on [Quantum Hub](https://quantum-hub.baidu.com). First, you should log into [Quantum Hub](https://quantum-hub.baidu.com), then enter the "User Management -> Feedback" page, and input the necessary information (choose "Get More Credit Points"). Submit your feedback and wait for a reply.

**Q: How should I cite Quanlse in my research?**  

**A:** We encourage developers to use Quanlse to do research & development in the field of quantum control. Please cite us by including [BibTeX file](Quanlse.bib).

## Changelog

The changelog of this project can be found in [CHANGELOG.md](CHANGELOG.md).

## Copyright and License

Quanlse uses [Apache-2.0 license](LICENSE).

## References

[1] [Quantum Computing - Wikipedia](https://en.wikipedia.org/wiki/Quantum_computing).

[2] Nielsen, Michael A., and Isaac L. Chuang. *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge: Cambridge UP, 2010. Print.

[3] [Werschnik, J., and E. K. U. Gross. "Quantum optimal control theory." *Journal of Physics B: Atomic, Molecular and Optical Physics* 40.18 (2007): R175.](https://doi.org/10.1088/0953-4075/40/18/R01)

[4] [Wendin, Göran. "Quantum information processing with superconducting circuits: a review." *Reports on Progress in Physics* 80.10 (2017): 106001.](https://doi.org/10.1088/1361-6633/aa7e1a)

[5] [Krantz, Philip, et al. "A quantum engineer's guide to superconducting qubits." *Applied Physics Reviews* 6.2 (2019): 021318.](https://doi.org/10.1063/1.5089550)


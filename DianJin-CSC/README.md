<div align="center">
    <h1><b>Evaluating, Synthesizing, and Enhancing for Customer Support Conversation</b></h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.04423-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.04423)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/aliyun/qwen-dianjin)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Dataset-orange.svg)](https://huggingface.co/DianJin)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-orange.svg)](https://modelscope.cn/organization/tongyi_dianjin)

[**中文**](README_zh.md) | **EN**

</div>

## Table of Contents
- [Introduction](#summary)
- [Dataset Download](#download)
- [Performance Comparison](#performance)
- [License](#license)
- [Citation](#cite)

## 📢 Introduction<a name="summary"></a>

![example.png](./images/example.png)

Effective customer support requires not only accurate problem-solving but also structured and empathetic communication aligned with professional standards. However, existing dialogue datasets often lack strategic guidance, and realworld service data is difficult to access and annotate. To address this, we introduce the task of Customer Support Conversation (CSC), aimed at training customer service supporters to respond using well-defined support strategies. We propose a structured CSC framework grounded in COPC guidelines, defining five conversational stages and twelve strategies to guide high-quality interactions. Based on this, we
construct CSConv, an evaluation dataset of 1,855 real-world customer–agent conversations rewritten using LLMs to reflect deliberate strategy use, and annotated accordingly. Additionally, we develop a role-playing approach that simulates strategy-rich conversations using LLM-powered roles
aligned with the CSC framework, resulting in the training dataset RoleCS. Experiments show that fine-tuning strong
LLMs on RoleCS significantly improves their ability to generate high-quality, strategy-aligned responses on CSConv.
Human evaluations further confirm gains in problem resolution

## 📥 Dataset Download<a name="download"></a>

|        |                        ModelScope                         |               HuggingFace               |
|:------:|:---------------------------------------------------------:|:---------------------------------------:|
| CSConv | [Data](https://modelscope.cn/organization/tongyi_dianjin) | [Data](https://huggingface.co/DianJin/) |
| RoleCS | [Data](https://modelscope.cn/organization/tongyi_dianjin) | [Data](https://huggingface.co/DianJin/) |

## 📊 Performance Comparison<a name="performance"></a>

![eval.png](./images/eval.png)

## 📋 License<a name="license"></a>
![](https://img.shields.io/badge/License-MIT-blue.svg#id=wZ1Hr&originHeight=20&originWidth=82&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
This project adheres to [MIT License](https://lbesson.mit-license.org/).

## 🔖 Citation<a name="cite"></a>

If you use our dataset, please cite our paper.

```
@article{dianjin-csc,
    title   = {Evaluating, Synthesizing, and Enhancing for Customer Support Conversation}, 
    author  = {Jie Zhu, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang, and Fang Kong},
    journal = {arxiv},
    year    = {2025}
}
```

<div align="center">
    <img src="images/dianjin_logo.png" alt="DianJin Logo" style="width: 200px;">
    <p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
        💜 <a href="https://tongyi.aliyun.com/dianjin" target="_blank" style="margin-left: 10px">Qwen DianJin Platform</a>  •
        🤗 <a href="https://huggingface.co/DianJin" target="_blank" style="margin-left: 10px">HuggingFace</a>  • 
        🤖 <a href="https://modelscope.cn/organization/tongyi_dianjin" target="_blank" style="margin-left: 10px">ModelScope</a> 
    </p>

[**中文**](README_zh.md) | **EN**

</div>

## 🚀 News 
- **2025.08.08** 🔥🔥🔥 "[Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](https://arxiv.org/abs/2508.04423)" is now published and open source!
- **2025.05.22** 🔥🔥🔥 "[M<sup>3</sup>FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](https://arxiv.org/abs/2506.02510)" has been officially accepted by ACL-2025! 
- **2025.04.23** [DianJin-R1](DianJin-R1/README.md) series open source release! This release includes the DianJin-R1-Data dataset, as well as two powerful models: DianJin-R1-7B and DianJin-R1-13B. Please check out our technical report "[DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models](https://arxiv.org/abs/2504.15716)" for more details and explore the capabilities of these new models.
- **2025.01.06** The [CFLUE](https://github.com/aliyun/cflue) dataset has been fully open-sourced and is now available for download!  🚀🚀🚀
- **2024.05.16** The paper "[Benchmarking Large Language Models on CFLUE - A Chinese Financial Language Understanding Evaluation Dataset](https://arxiv.org/abs/2405.10542)" has been officially accepted by ACL-2024! 🚀🚀🚀

The **data** and **models** that have been released so far are as follows:

<table style="width: 100%; text-align: center;">
    <tr>
        <td></td>
        <th>ModelScope</th>
        <th>HuggingFace</th>
        <th>Paper</th>
    </tr>
    <tr>
        <th>CSC</th>
        <td><a href="https://www.modelscope.cn/datasets/tongyi_dianjin/DianJin-CSC-Data">CSC</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/DianJin-CSC-Data">CSC</a></td>
        <td><a href="https://arxiv.org/abs/2508.04423">Paper</a></td>
    </tr>
    <tr>
        <th>M<sup>3</sup>FinMeeting</th>
        <td colspan="2">Releasing Soon</td>
        <td><a href="https://arxiv.org/abs/2506.02510">ACL-2025</a></td>
    </tr>
    <tr>
        <th rowspan="3">DianJin-R1</th>
        <td><a href="https://www.modelscope.cn/models/tongyi_dianjin/DianJin-R1-32B">DianJin-R1-32B</a></td>
        <td><a href="https://huggingface.co/DianJin/DianJin-R1-32B">DianJin-R1-32B</a></td>
        <td rowspan="3"><a href="https://arxiv.org/abs/2504.15716">Technical Report</a></td>
    </tr>
    <tr>
        <td><a href="https://www.modelscope.cn/models/tongyi_dianjin/DianJin-R1-7B">DianJin-R1-7B</a></td>
        <td><a href="https://huggingface.co/DianJin/DianJin-R1-7B">DianJin-R1-7B</a></td>
    </tr>
    <tr>
        <td><a href="https://www.modelscope.cn/datasets/tongyi_dianjin/DianJin-R1-Data">DianJin-R1-Data</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/DianJin-R1-Data">DianJin-R1-Data</a></td>
    </tr>
    <tr>
        <th>CFLUE</th>
        <td><a href="https://modelscope.cn/datasets/tongyi_dianjin/CFLUE">CFLUE</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/CFLUE">CFLUE</a></td>
        <td><a href="https://arxiv.org/abs/2405.10542">ACL-2024</a></td>
    </tr>
</table>

## 📝 Introduction

Welcome to Qwen DianJin 👋

Tongyi DianJin is a financial intelligence solution platform built by Alibaba Cloud, 
dedicated to providing financial business developers with a convenient artificial intelligence application development environment. 
We not only focus on launching advanced large language models (LLM) and large multimodal models (LMM), but also serve as a financial assistant that integrates various artificial intelligence technologies. 
Through our platform, you can explore and experience innovative applications related to artificial general intelligence (AGI), driving development and innovation in the financial sector.

We welcome you to explore and experience, and together embark on a journey of intelligent finance!

## ✨ Features

### 💡 Intelligent Applications

Provide standardized API capabilities for financial scenarios, such as research report summarization, information extraction from news, and intent recognition for financial customer service.

- ✅ Financial Services: Such as credit card repayment reminders, mobile banking navigation, renewal prompts, marketing material generation, etc.
- ✅ Investment Research & News: Such as research report summarization, information extraction, financial translation, trading metrics Q&A, etc.
- ✅ Operational Data Query: Such as operational metrics Q&A, anomaly alerts, and other intelligent operational capabilities.
- ...

### 💡 Open Platform

Equip developers with a suite of financial APIs and tools, making it easy to integrate and extend functionality.

- ✅ Document Q&A: Optimized document parsing and recall ranking strategies, providing knowledge base Q&A capabilities tailored for financial scenarios.
- ✅ Metrics Q&A: Capable of answering questions about metrics and plotting metrics, enhancing understanding of financial expertise.
- ✅ Multi-Agent System: Includes configuration and orchestration of various types of nodes, supporting more personalized configurations based on the capabilities provided by DianJin.
- ...

## 🔖 Citation

If you find our work helpful, feel free to give us a cite.

```
@article{csc,
    title = {Evaluating, Synthesizing, and Enhancing for Customer Support Conversation}, 
    author = {Jie Zhu, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang, Fang Kong},
    journal = {https://arxiv.org/abs/2508.04423},
    year = "2025"
}

@inproceedings{m3finmeeting,
    title = "M^{3}FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset",
    author = "Jie Zhu, Junhui Li, Yalong Wen, Xiandong Li, Lifan Guo, Feng Chen",
    booktitle = "Findings of ACL",
    year = "2025"
}

@article{dianjin-r1,
    title = {DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models}, 
    author = {Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang},
    journal = {arxiv.org/abs/2504.15716},
    year = "2025"
}

@inproceedings{cflue,
    title = "Benchmarking Large Language Models on CFLUE - A Chinese Financial Language Understanding Evaluation Dataset",
    author = "Jie Zhu, Junhui Li, Yalong Wen, Lifan Guo",
    booktitle = "Findings of ACL",
    year = "2024",
    pages = "5673--5693",
}
```

## 🤝 Contact Us

Thank you very much for your interest in the Tongyi Dianjin series! 
If you would like to leave a message for our research or product team, feel free to contact us via our official email or by scanning the code to join our DingTalk group: CFLUE@alibabacloud.com. 
Our team is committed to providing you with assistance and support.

<img src="images/dianjin_dingding.png" alt="DianJin Logo" style="width: 200px;">


## ⚠️ Disclaimer

We assume no legal liability for the use of the DianJin open-source model and data. Users are responsible for independently assessing and assuming any potential risks associated with using the DianJin model or data, and should always exercise caution.
We recommend that users independently verify and analyze the model's outputs, and make informed decisions based on their specific needs and real-world scenarios.
By providing open-source data and models, we aim to offer valuable tools for academic research and industry applications, promoting advancements in artificial intelligence technology within data analysis, financial innovation, and other related fields.
We encourage users to fully leverage their creativity, deeply explore the potential of the DianJin model, expand its application scenarios, and collectively drive progress and practical implementation of AI technologies across various domains.

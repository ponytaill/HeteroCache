<div >

# HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference

<p>
  <a href="https://arxiv.org/abs/2601.13684">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Paper">
  </a>
</p>

## 📖 Abstract
The linear memory growth of the KV cache poses a significant bottleneck for LLM inference in long-context tasks. Existing static compression methods often fail to preserve globally important information, principally because they overlook the attention drift phenomenon where token significance evolves dynamically. Although recent dynamic retrieval approaches attempt to address this issue, they typically suffer from coarse-grained caching strategies and incur high I/O overhead due to frequent data transfers. To overcome these limitations, we propose HeteroCache, a training-free dynamic compression framework. Our method is built on two key insights: attention heads exhibit diverse temporal heterogeneity, and there is significant spatial redundancy among heads within the same layer.Guided by these insights, HeteroCache categorizes heads based on stability and redundancy.
Consequently, we apply a fine-grained weighting strategy that allocates larger cache budgets to heads with rapidly shifting attention to capture context changes, thereby addressing the inefficiency of coarse-grained strategies.Furthermore, we employ a hierarchical storage mechanism in which a subset of representative heads monitors attention shift, and trigger an asynchronous, on-demand retrieval of contexts from the CPU, effectively hiding I/O latency.
Finally, experiments demonstrate that HeteroCache achieves state-of-the-art performance on multiple long-context benchmarks and accelerates decoding by up to $3\times$ compared to the original model in the 224K context. Our code will be open-source.

## 🚧 Coming Soon

The source code for **HeteroCache** is currently being prepared and will be released soon. 

## 📌 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{shi2026heterocache,
  title={HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference},
  author={Shi, Zhiyuan and Qiu, Qibo and Xue, Feng and Jiang, Zhonglin and Yu, Li and Jiang, Jian and He, Xiaofei and Wang, Wenxiao},
  journal={arXiv preprint arXiv:2601.13684},
  year={2026}
}

<h1 align='center'>V²Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation</h1>




<div align='center'>
    <a href='https://arxiv.org/abs/2503.07493'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

<!-- ## 🚀 Overview
<p align="center">
  <img src="assets/overview.png" width="720p">
</p> -->

# 📖 Introduction

V²Flow introduces an advanced vector-quantized image tokenizer designed to seamlessly integrate visual tokenization with existing large language model (LLM) vocabularies. By aligning structural representations and latent distributions between image tokens and textual tokens, V²Flow enables effective autoregressive image generation leveraging pre-trained LLMs.

# ✨ Highlights

###  1. Structural and Latent Distribution Alignment  with LLM's Vocabulary:



<p align="center">
<img src="./assets/visual_vocabulary_resampler.png" width=100%>
<p>

###  2. Masked Autoregressive Reconstruction from a Flow-matching Perspective:
<p align="center">
<img src="assets/masked_autoregressive_decoder.png" width=100%>
<p>


### 3. Autoregressive Visual Generation on Top of Existing LLMs:


<p align="center">
<img src="assets/generation_results.png" width=100%>
<p>






# 🧩 Project Updates
* **2025-03-31:** Release of the complete training and inference codebase for [V²Flow](https://arxiv.org/abs/2503.07493). Pretrained models (1024x1024 and 512x512 resolutions) will be available shortly.
* **2025-03-10:** [V²Flow](https://arxiv.org/abs/2503.07493) is released on arXiv.

# 🚀 Training & Inference

## V²Flow Tokenizer
The complete data preparation, training, and inference instructions for the V²Flow tokenizer can be found [here](docs/V2Flow.md).




# 🚀 Open-source Plan

- V²Flow tokenizer
  - [x] Training and inference codes 
  - [ ] Checkpoints
  - [ ] Gradio Demo
- V²Flow+LLaMA for Autoregressive Visual Generation
  - [ ] Training and inference codes  
  - [ ] Checkpoints
  - [ ] Gradio Demo



# Acknowledgement

We thank the great work from [MAR](https://github.com/LTH14/mar),  [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoLLaMA](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
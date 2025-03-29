### Preparation

#### Step 1: Pre-computing DC-AE Latents

To accelerate convergence during training, it is strongly recommended to pre-compute and cache the [DC-AE](https://github.com/mit-han-lab/efficientvit/blob/master/assets/docs/dc_ae_sana_1.1.md) latents. This significantly reduces redundant computations during the training of the V²Flow tokenizer.

Execute the following command:

```shell
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_cache.py \
  --img_size 1024 \
  --vae_path mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
  --vae_embed_dim 32 \
  --batch_size 32 \
  --data_path $DATASET_PATH \
  --cached_path $CACHE_PATH
```

#### Step 2: Caching Pretrained LLM Vocabulary

The following Python script demonstrates how to efficiently load and cache the pretrained LLM vocabulary weights:

```python
import os
import torch
from safetensors import safe_open

def load_safetensors_from_directory(directory):
    tensors = {}
    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    return tensors

loaded_tensors = load_safetensors_from_directory($PRETRAINED_LLM_Vocabulary)

# Save embed_tokens weights separately
for key, tensor in loaded_tensors.items():
    if 'embed_tokens' in key:
        print(f"Key: {key}, Tensor shape: {tensor.shape}")
        torch.save(tensor, $PRETRAINED_LLM_Vocabulary + '.pth')
```

### Training

The V²Flow tokenizer supports diverse training configurations, including different flow-matching strategies and the option to integrate pretrained LLM vocabularies. Below is an example command for training a V²Flow tokenizer at a resolution of 1024x1024:

```shell
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=0 \
  --master_addr=$hostname --master_port=$PORT \
  main_v2flow.py \
  --img_size 1024 \
  --visual_vocabulary True \
  --visual_codebook_size 16384 \
  --flow_method linear_flow \
  --llm_codebook_path $PRETRAINED_LLM_Vocabulary
```

The table below details essential training arguments:

| Argument | Description |
|:---|:---|
| `--visual_vocabulary` | Type: bool. If set to `False`, visual tokens will not utilize embeddings from the pretrained LLM vocabulary. |
| `--visual_codebook_size` | Integer specifying the size of the visual vocabulary codebook. |
| `--llm_codebook_path` | Path to the pretrained LLM vocabulary pth. |
| `--flow_method` | Specifies the flow-matching strategy used during training. Supported methods include [`CondOT`](https://github.com/facebookresearch/flow_matching) from Facebook Research and [`linear_flow`](https://github.com/huggingface/diffusers) from Hugging Face Diffusers. Empirical evaluations suggest that `linear_flow` achieves higher computational efficiency, generating images at 1024x1024 resolution within approximately 2 seconds on an A800 GPU, compared to about 5 seconds required by CondOT. |


### Inference
First, please follow Preparation Step 1 to cache the latents of test images,  saving them to the $TEST_CACHE_DIR. Then, execute the inference command as follows:

```shell
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=0 \
  --master_addr=$hostname --master_port=$PORT \
  main_v2flow.py \
  --img_size 1024 \
  --visual_vocabulary True \
  --visual_codebook_size 16384 \
  --flow_method linear_flow \
  --llm_codebook_path $PRETRAINED_LLM_Vocabulary \
  --test_cache_dir $TEST_CACHE_DIR \
  --evaluate 
```
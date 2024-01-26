# CCA: Collaborative Competitive Agents for Image Editing

[Paper](https://arxiv.org/abs/2401.13011) | [Code](https://github.com/TiankaiHang/CCA)

## Abstract

> This paper presents a novel generative model, Collaborative Competitive Agents (CCA), which leverages the capabilities of multiple Large Language Models (LLMs) based agents to execute complex tasks. Drawing inspiration from Generative Adversarial Networks (GANs), the CCA system employs two equal-status generator agents and a discriminator agent. The generators independently process user instructions and generate results, while the discriminator evaluates the outputs, and provides feedback for the generator agents to further reflect and improve the generation results. Unlike the previous generative model, our system can obtain the intermediate steps of generation. This allows each generator agent to learn from other successful executions due to its transparency, enabling a collaborative competition that enhances the quality and robustness of the system's results. The primary focus of this study is image editing, demonstrating the CCA's ability to handle intricate instructions robustly. The paper's main contributions include the introduction of a multi-agent-based generative model with controllable intermediate steps and iterative optimization, a detailed examination of agent relationships, and comprehensive experiments on image editing.

![framework](https://github.com/TiankaiHang/storage-2023/releases/download/research/cca-framework.png)

## Preparation

Please refer to the [installation guide](./assets/installation.md), which provides detailed instructions from setting up checkpoints and organizing datasets to the installation of required packages.

## Running the code

You should set the `OPENAI_API_KEY` first. If you use the key from OpenAI, you can set the environment variables as follows:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
```

If you use the ChatGPT service at [Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython&pivots=programming-language-studio), you can set the environment variables as follows:

```bash
export OPENAI_API_TYPE=azure
export OPENAI_API_VERSION=YOUR_API_VERSION
export OPENAI_API_BASE=YOUR_API_BASE
export OPENAI_API_KEY=YOUR_API_KEY
```

Then you can run the code as follows:


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/main.py run \
    --image-path assets/cute.jpg \
    --instruction "Rotate the image counterclockwise and then add a pair of glasses to the cat." \
    --num-agents 2 \
    --num-rounds 3 \
    --tag TAG-temp0.8-multitool-round3 \
    --tool-list InstructDiffusion,EnhanceColor,GaussianBlur,RGB2Gray,RotateClockwise,RotateCounterClockwise
```


`CUDA_VISIBLE_DEVICES` is used to specify the visible GPU device. `--image-path` specifies the input image. `--instruction` is the editing instruction. `--num-agents` is the number of agents, default is 2. `--tag` is the tag of the experiment. `--num-rounds` is the number of rounds. `--tool-list` is the list of tools. The tools are separated by `,`. You could refer to [`src/tools`](./src/tools.py) for the available tools.

## Citation

If you find our work useful for your research, please consider citing our paper. 😊

```bibtex
@article{cca2023,
  title={CCA: Collaborative Competitive Agents for Image Editing},
      author={Tiankai Hang and Shuyang Gu and Dong Chen and Xin Geng and Baining Guo},
      year={2024},
      eprint={2401.13011},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
```

## Acknowlegements

This repository is inspired by [InstructDiffusion](https://github.com/cientgu/InstructDiffusion), [ChatDev](https://github.com/OpenBMB/ChatDev), and [TaskMatrix](https://github.com/moymix/TaskMatrix/tree/main).
We also utilize awesome tools from [LLaVA](https://github.com/haotian-liu/LLaVA), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [EDICT](https://github.com/salesforce/EDICT), and [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).

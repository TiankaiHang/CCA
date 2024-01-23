# Installation

Our code is tested on `python=3.10`, `CUDA=11.8` and `torch==2.0.1`. 
It is recommended to use `CUDA` with `11.3` or higher version. You can check your `CUDA` version by `nvcc -V`.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Prepare checkpoints

You can download checkpoints of `InstructDiffusion`, `InstructPix2Pix`, and `MagicBrush` through

```bash
bash scripts/download_instructdiffusion.sh
```

You can download the checkpoints required by GroundingDINO through

```bash
bash scripts/download_groundingdino.sh
```


## Install required packages

### Common packages

```bash
pip install -r requirements.txt
```

### GroundingDINO

```bash
pip install segment_anything
git clone https://github.com/IDEA-Research/GroundingDINO.git src/GroundingDINO # make sure nvcc -V >= 11.3
pip install -e src/GroundingDINO
```

### LLaVA

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA/
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install ninja
pip install flash-attn --no-build-isolation
cd ..
```

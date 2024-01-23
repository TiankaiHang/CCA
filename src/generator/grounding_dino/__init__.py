r"""
```bash

export PATH=/usr/local/cuda-11.8/bin:$PATH

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O checkpoints/groundingdino_swint_ogc.pth

pip install segment_anything
git clone https://github.com/IDEA-Research/GroundingDINO.git -d src/GroundingDINO
cd GroundingDINO
pip install -e .
```
"""
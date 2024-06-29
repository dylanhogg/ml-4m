from datetime import datetime
import os
from fourm.demo_4M_sampler import Demo4MSampler, img_from_url

PYTORCH_ENABLE_MPS_FALLBACK = os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', None)
DEVICE = "mps"
MODEL = "EPFL-VILAB/4M-7_B_CC12M"  # 445M params, F32 type, 8,725 downloads
# MODEL = "EPFL-VILAB//4M-21_XL"  # 4.54B params, F32 type, 484 downloads (default demo model)

# MODS = None  # Default: ['det', 'tok_rgb@224', 'rgb@224', 'tok_semseg@224', 'tok_normal@224', 'tok_clip@224', 'caption', 'tok_depth@224']
# MODS_SR = None  # Default: ['det', 'tok_semseg@448', 'rgb@224', 'tok_normal@448', 'tok_clip@224', 'rgb@448', 'caption', 'tok_depth@448', 'tok_rgb@224', 'tok_semseg@224', 'tok_clip@448', 'tok_rgb@448', 'tok_normal@224', 'tok_depth@224']
MODS = ['caption']
MODS_SR = ['caption']

print(f"{DEVICE=}\n{MODEL=}\n{PYTORCH_ENABLE_MPS_FALLBACK=}")

if DEVICE == "mps" and PYTORCH_ENABLE_MPS_FALLBACK is None:
    raise ValueError("Set PYTORCH_ENABLE_MPS_FALLBACK=1 to use MPS device")

start = datetime.now()

sampler = Demo4MSampler(fm=MODEL, mods=MODS, mods_sr=MODS_SR).to(device=DEVICE)
img = img_from_url('https://storage.googleapis.com/four_m_site/images/demo_rgb.png') # 1x3x224x224 ImageNet-standardized PyTorch Tensor
preds = sampler({'rgb@224': img.to(device=DEVICE)}, seed=None) 

took = datetime.now() - start
print(f"took {took.total_seconds()}s on {DEVICE=}")

sampler.plot_modalities(preds, save_path=None)


"""Default pretrained checkpoint locations inside this repository."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOMO_CKPT_DIR = os.path.join(REPO_ROOT, 'ImageAlignment', 'Codes', 'checkpoints_homo')
RECON_CKPT_DIR = os.path.join(REPO_ROOT, 'ImageReconstruction', 'Codes', 'checkpoints')
HOMO_CKPT_STEP = '1000000'
RECON_CKPT_STEP = '200000'

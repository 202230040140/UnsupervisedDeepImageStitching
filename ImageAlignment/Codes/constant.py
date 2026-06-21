import os

_CODES_DIR = os.path.dirname(os.path.abspath(__file__))

# All paths can be overridden via environment variables so the reproduction
# orchestrator (reproduce/run_udis.py) can drive the same scripts for both
# UDIS-D and StitchBench without editing this file.

#training dataset path
TRAIN_FOLDER = os.environ.get('UDIS_TRAIN_FOLDER', '/data/cylin/nl/Data/UDIS-D/training')

#testing dataset path
TEST_FOLDER = os.environ.get('UDIS_TEST_FOLDER', '/data/cylin/nl/Data/UDIS-D/testing')

#GPU index
GPU = os.environ.get('UDIS_GPU', '0')

#batch size for training
TRAIN_BATCH_SIZE = 4

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 600000

# checkpoints path (folder holding model.ckpt-* of the Stage-1 homography model)
SNAPSHOT_DIR = os.environ.get('UDIS_HOMO_CKPT_DIR', os.path.join(_CODES_DIR, 'checkpoints_homo'))

# checkpoint step suffix, i.e. model.ckpt-<STEP>
HOMO_CKPT_STEP = os.environ.get('UDIS_HOMO_CKPT_STEP', '1000000')

#sumary path
SUMMARY_DIR = "./summary"

# where Stage-1 inference.py writes per-pair PSNR/SSIM (JSON). Empty -> skip.
METRICS_OUT = os.environ.get('UDIS_METRICS_OUT', '')

# where output_inference.py writes warp1/warp2/mask1/mask2 sub-folders.
WARP_OUT = os.environ.get('UDIS_WARP_OUT', '../output/testing')

# optional cap on the number of processed pairs (0 = all). Used for smoke tests.
LIMIT = int(os.environ.get('UDIS_LIMIT', '0'))

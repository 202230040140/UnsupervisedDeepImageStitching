import os

_CODES_DIR = os.path.dirname(os.path.abspath(__file__))

# All paths can be overridden via environment variables so the reproduction
# orchestrator (reproduce/run_udis.py) can drive the same scripts for both
# UDIS-D and StitchBench without editing this file.

#training dataset path
TRAIN_FOLDER = os.environ.get('UDIS_RECON_TRAIN_FOLDER', '../../ImageAlignment/output/training')

#testing dataset path (folder holding warp1/warp2/mask1/mask2 produced by Stage 1)
TEST_FOLDER = os.environ.get('UDIS_RECON_TEST_FOLDER', '../../ImageAlignment/output/testing')

#GPU index
GPU = os.environ.get('UDIS_GPU', '0')

#batch size for training
TRAIN_BATCH_SIZE = 1

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 200000

# checkpoints path (folder holding model.ckpt-* of the Stage-2 reconstruction model)
SNAPSHOT_DIR = os.environ.get('UDIS_RECON_CKPT_DIR', os.path.join(_CODES_DIR, 'checkpoints'))

# checkpoint step suffix, i.e. model.ckpt-<STEP>
RECON_CKPT_STEP = os.environ.get('UDIS_RECON_CKPT_STEP', '200000')

#sumary path
SUMMARY_DIR = "./summary"

# where inference.py writes the stitched panoramas.
RESULT_OUT = os.environ.get('UDIS_RESULT_OUT', '../results')

# optional cap on the number of processed pairs (0 = all). Used for smoke tests.
LIMIT = int(os.environ.get('UDIS_LIMIT', '0'))

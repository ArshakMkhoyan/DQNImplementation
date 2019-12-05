import os

base_dir = './results'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


checkpoint_dir = os.path.join(base_dir, "checkpoints")
checkpoint_save_path = os.path.join(checkpoint_dir, "my_model.h5")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

summary_dir = os.path.join(base_dir, "summaries")
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

IMG_SIZE = (84, 84, 1)
N_FRAMES = 4
MAKE_GREY = True
CROP_FUNC = lambda x: x[30:203, 6:154, :]

GAMMA = 0.99
GAME_NAME = 'BreakoutDeterministic-v4' # 'SpaceInvadersDeterministic-v4'
DOUBLE_DQN = True
CLIP_DELTA = 1.0
HUBER_LOSS = True
BATCH_SIZE = 32
MAX_FRAMES = 50 * 10 ** 6
EXP_BUFFER_SIZE = 650000
EXP_BUFFER_START_SIZE = 5 * 10 ** 4
UPDATE_NET_N_FRAMES = 4
UPDATE_TARGET_NET_N_FRAMES = 10 ** 4
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 10 ** 6
SESSIONS_TO_EVALUATE = 10
SAVE_MODEL_FREQ = 10 ** 6

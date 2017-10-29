
BATCH_SIZE = 512
SEQ_LEN = 30
PREDICT_LEN = 1
NUM_LAYERS = 2
HIDDEN_UNITS = 128
# FC_NUM_OUTPUT = 16
NUM_CLASSES = 20
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT = 0.1
EPOCH = 30
# EXAMPLES = 5e6
EXAMPLES = 4030508
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 7
CELL_TYPE = 'RNN'
# FILEDS = ['open', 'close', 'high', 'low', 'volume']
# IDX = 1
# DATA_PATH = '/home/daiab/machine_disk/code/Craft/rnn/share/all_data.pkl.2015-01-05_2017-09-13'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/code/quantum/photon/ohlcvr_ratio_norm.rec'
# INFER_DATA_PATH = '/home/daiab/machine_disk/code/quantum/utils/val_data.pkl'
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/models/gluon_cls/save/model'


# infer
INFER_SIZE = 10


BATCH_SIZE = 512
SEQ_LEN = 30
PREDICT_LEN = 1
NUM_LAYERS = 2
HIDDEN_UNITS = 128
# FC_NUM_OUTPUT = 16
NUM_CLASSES = 41
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT_KEEP = 0.9
EPOCH = 30
# EXAMPLES = 5e6
EXAMPLES = 4699085
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
# INPUT_SIZE = 5
CELL_TYPE = 'RNN'
# FILEDS = ['open', 'close', 'high', 'low', 'volume']
# IDX = 1
# DATA_PATH = '/home/daiab/machine_disk/code/Craft/rnn/share/all_data.pkl.2015-01-05_2017-09-13'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/code/quantum/photon/ohlcvr_ratio_norm.records'
# INFER_DATA_PATH = '/home/daiab/machine_disk/code/quantum/utils/val_data.pkl'
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/models/atom_cls/save/model'


# infer
INFER_SIZE = 10

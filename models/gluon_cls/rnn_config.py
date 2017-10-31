
BATCH_SIZE = 512
SEQ_LEN = 30
PREDICT_LEN = 1
NUM_LAYERS = 5
HIDDEN_UNITS = 380
# FC_NUM_OUTPUT = 16
NUM_CLASSES = 20
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT = 0.1
EPOCH = 60
# EXAMPLES = 5e6
EXAMPLES = 4030508  # for 20 classes
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 7
CELL_TYPE = 'rnn_tanh'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/code/quantum/photon/ohlcvr_ratio_norm.rec'
# INFER_DATA_PATH = ''
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/models/gluon_cls/save/mx_model.params'


# infer
INFER_SIZE = 10

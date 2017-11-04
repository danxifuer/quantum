
BATCH_SIZE = 512
SEQ_LEN = 50
PREDICT_LEN = 1
NUM_LAYERS = 3
HIDDEN_UNITS = 128
# FC_NUM_OUTPUT = 16
NUM_CLASSES = 2
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT = 0.1
EPOCH = 60
# EXAMPLES = 5e6
EXAMPLES = 2794086
ITER_NUM_EPCOH = int(EXAMPLES / BATCH_SIZE)
DECAY_STEP = ITER_NUM_EPCOH * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 7
CELL_TYPE = 'rnn_relu'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/code/quantum/ohlcvr_from_norm_data_50_len_2_cls.rec'
# INFER_DATA_PATH = ''
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/models/gluon_cls_use_across_data/save/mx_model.params'


# infer
INFER_SIZE = 10

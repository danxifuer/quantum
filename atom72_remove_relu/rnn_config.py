
BATCH_SIZE = 512
SEQ_LEN = 30
PREDICT_LEN = 2
NUM_LAYERS = 2
HIDDEN_UNITS = 128
FC_NUM_OUTPUT = 16
# INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
ATTN_LENGTH = 10
DROPOUT_KEEP = 0.9
EPOCH = 60
# EXAMPLES = 5e6
EXAMPLES = 5162363
DECAY_STEP = EXAMPLES / BATCH_SIZE * EPOCH
LR = 0.04
END_LR = 0.0002
INPUT_SIZE = 5
CELL_TYPE = 'RNN'
FILEDS = ['open', 'close', 'high', 'low', 'volume']
IDX = 1
# DATA_PATH = '/home/daiab/machine_disk/code/Craft/rnn/share/all_data.pkl.2015-01-05_2017-09-13'
TRAIN_DATA_PATH = '/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl'
INFER_DATA_PATH = '/home/daiab/machine_disk/code/quantum/utils/val_data.pkl'
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/atom72_remove_relu/save/model'


# infer
INFER_SIZE = 10


BATCH_SIZE = 512
SEQ_LEN = 25
PREDICT_LEN = 10
NUM_LAYERS = 5
HIDDEN_UNITS = 256
FC_NUM_OUTPUT = 48
INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
EPOCH = 20
# EXAMPLES = 5e6
EXAMPLES = 5162363
DECAY_STEP = EXAMPLES / BATCH_SIZE * EPOCH
LR = 0.04
END_LR = 0.00005
INPUT_SIZE = 6
CELL_TYPE = 'RNN'
FILEDS = ['open', 'close', 'high', 'low', 'total_turnover', 'volume']
IDX = 1
# DATA_PATH = '/home/daiab/machine_disk/code/Craft/rnn/share/all_data.pkl.2015-01-05_2017-09-13'
DATA_PATH = '/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl'
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/atom72_remove_relu/save/model'


# infer
INFER_SIZE = 10

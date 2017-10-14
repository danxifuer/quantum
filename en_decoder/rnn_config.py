

BATCH_SIZE = 512
SEQ_LEN = 25
PREDICT_LEN = 3
NUM_LAYERS = 1
HIDDEN_UNITS = 512
FC_NUM_OUTPUT = 96
INPUT_FC_NUM_OUPUT = 16
NUM_RESIDUAL_LAYERS = NUM_LAYERS - 1
EPOCH = 20
# EXAMPLES = 5e6
EXAMPLES = 5162363
DECAY_STEP = EXAMPLES / BATCH_SIZE * EPOCH
LR = 0.04
END_LR = 0.0001
INPUT_SIZE = 6
CELL_TYPE = 'RNN'
FILEDS = ['open', 'close', 'high', 'low', 'total_turnover', 'volume']
IDX = 1
# DATA_PATH = '/home/daiab/machine_disk/code/Craft/rnn/share/all_data.pkl.2015-01-05_2017-09-13'
DATA_PATH = '/home/daiab/machine_disk/data/tushare_data/rq_1d.pkl'
RESTORE_PATH = '/home/daiab/machine_disk/code/quantum/en_decoder/save/model'

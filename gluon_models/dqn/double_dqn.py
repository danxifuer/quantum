from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn, rnn
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryFile


class Options:
    def __init__(self):
        # architecture
        self.batch_size = 32
        self.num_actions = 9
        self.seq_len = 500
        self.num_layers = 5
        self.num_inputs = 5
        self.num_hidden = 128
        self.mode = 'rnn_tanh'
        self.csv = '/home/daiab/machine_disk/code/quantum/database/RB_min.csv'
        # trick
        self.replay_buffer_size = 1000000
        self.learning_frequency = 1
        self.skip_frame = 4
        self.internal_skip_frame = 4
        self.frame_len = 4
        self.target_update = 10000
        self.epsilon_min = 0.1
        self.annealing_end = 1000000.
        self.gamma = 0.99
        self.replay_start_size = 5000
        self.no_op_max = 30 / self.skip_frame
        # optimization
        self.num_episode = 150
        self.lr = 0.00025
        self.gamma1 = 0.95
        self.gamma2 = 0.95
        self.rms_eps = 0.01
        self.ctx = mx.gpu()


def read_csv(csv_file):
    rb = pd.read_csv(csv_file, index_col=0)
    rb.index = pd.DatetimeIndex(rb.index)
    close = (rb.loc[:, 'close'] - rb.loc[:, 'close'].rolling(window=5).mean()) / \
            rb.loc[:, 'close'].rolling(window=5).std()
    nan = close.isnull()
    rb = rb.loc[~nan, :]
    close = close[~nan]
    return np.array(rb), np.array(close)


opt = Options()
manual_seed = 1
mx.random.seed(manual_seed)
attrs = vars(opt)
print(', '.join("%s: %s" % item for item in attrs.items()))


class Env:
    def __init__(self, csv):
        self._record = 0
        self._idx = 0
        self._seq_len = opt.seq_len
        self._data, self._target = read_csv(csv)
        self._size = self._data.shape[0]
        self._ratio = 10
        self._earn = 0

    def reset(self):
        self._idx = 0
        self._earn = 0
        self._record = 0
        return self._data[self._idx: self._idx + self._seq_len]

    def step(self, action):
        if action == 0:
            sb = 0
        elif action <= 4:
            sb = self._ratio * action
        elif action <= 8:
            sb = (self._ratio - 12) * action
        else:
            raise Exception('action error %s' % action)
        self._record += sb
        cur_reward = sb * self._target[self._idx]
        self._earn += cur_reward
        self._idx += 1
        has_done = False
        if self._idx >= (self._size - self._seq_len):
            has_done = True
        return self._data[self._idx: self._idx + self._seq_len], cur_reward, has_done


class RNNClsModel(gluon.Block):
    def __init__(self, dropout=0.0, **kwargs):
        super(RNNClsModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            # self.emb = nn.Embedding(vocab_size, num_embed,
            #                         weight_initializer=mx.init.Uniform(0.1))
            if opt.mode == 'rnn_relu':
                self.rnn = rnn.RNN(opt.num_hidden, activation='relu',
                                   num_layers=opt.num_layers, layout='NTC',
                                   dropout=dropout, input_size=opt.num_inputs)
            elif opt.mode == 'rnn_tanh':
                self.rnn = rnn.RNN(opt.num_hidden, num_layers=opt.num_layers,
                                   layout='NTC', dropout=dropout,
                                   input_size=opt.num_inputs)
            elif opt.mode == 'lstm':
                self.rnn = rnn.LSTM(opt.num_hidden, num_layers=opt.num_layers,
                                    layout='NTC', dropout=dropout,
                                    input_size=opt.num_inputs)
            elif opt.mode == 'gru':
                self.rnn = rnn.GRU(opt.num_hidden, num_layers=opt.num_layers,
                                   layout='NTC', dropout=dropout, input_size=opt.num_inputs)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % opt.mode)

            self.fc = nn.Dense(opt.num_actions, in_units=opt.num_hidden * opt.seq_len)
            self.num_hidden = opt.num_hidden
            self.seq_len = opt.seq_len

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.drop(output)
        decoded = self.fc(output.reshape((-1, self.num_hidden * self.seq_len)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


env = Env(opt.csv)

dqn = RNNClsModel()
dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
DQN_trainer = gluon.Trainer(dqn.collect_params(), 'RMSProp',
                            {'learning_rate': opt.lr, 'gamma1': opt.gamma1, 'gamma2': opt.gamma2,
                             'epsilon': opt.rms_eps, 'centered': True})
dqn.collect_params().zero_grad()

target_dqn = RNNClsModel()
target_dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


def rew_clipper(rew):
    if rew > 0.:
        return 1.
    elif rew < 0.:
        return -1.
    else:
        return 0


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


l2loss = gluon.loss.L2Loss(batch_axis=0)

frame_counter = 0.
annealing_count = 0.
epis_count = 0.
replay_memory = ReplayBuffer(opt.replay_buffer_size)
tot_clipped_reward = np.zeros(opt.num_episode)
tot_reward = np.zeros(opt.num_episode)
moving_average_clipped = 0.
moving_average = 0.

batch_state = nd.empty((opt.batch_size, opt.seq_len, opt.num_inputs), opt.ctx)
batch_state_next = nd.empty((opt.batch_size, opt.seq_len, opt.num_inputs), opt.ctx)

for i in range(opt.num_episode):
    cum_clipped_reward = 0
    cum_reward = 0
    state = env.reset()
    t = 0.
    done = False
    target_dqn_hidden = target_dqn.begin_state(func=mx.nd.zeros,
                                               batch_size=opt.batch_size,
                                               ctx=opt.ctx)
    dqn_hidden = dqn.begin_state(func=mx.nd.zeros,
                                 batch_size=opt.batch_size,
                                 ctx=opt.ctx)
    dqn_hidden_one = dqn.begin_state(func=mx.nd.zeros,
                                     batch_size=1,
                                     ctx=opt.ctx)

    while not done:
        previous_state = state
        sample = random.random()
        if frame_counter > opt.replay_start_size:
            annealing_count += 1
        if frame_counter == opt.replay_start_size:
            print('annealing and learning are started ')

        eps = np.maximum(1. - annealing_count / opt.annealing_end, opt.epsilon_min)
        effective_eps = eps
        if t < opt.no_op_max:
            effective_eps = 1.

        # epsilon greedy policy
        if sample < effective_eps:
            action = random.randint(0, opt.num_actions - 1)
        else:
            data = nd.array(state.reshape([1, opt.seq_len, opt.num_inputs]), opt.ctx)
            action = int(nd.argmax(dqn(data, dqn_hidden_one)[0], axis=1).as_in_context(
                mx.cpu()).asscalar())

        # Skip frame
        rew = 0
        for skip in range(opt.skip_frame - 1):
            next_frame, reward, done = env.step(action)
            cum_clipped_reward += rew_clipper(reward)
            rew += reward
            for internal_skip in range(opt.internal_skip_frame - 1):
                _, reward, done = env.step(action)
                cum_clipped_reward += rew_clipper(reward)
                rew += reward

        next_state, reward, done = env.step(action)
        cum_clipped_reward += rew_clipper(reward)
        rew += reward
        cum_reward += rew

        reward = rew_clipper(rew)
        replay_memory.push(previous_state, action, next_state, reward, done)
        # Train
        if frame_counter > opt.replay_start_size:
            if frame_counter % opt.learning_frequency == 0:
                print('training')
                transitions = replay_memory.sample(opt.batch_size)
                batch = Transition(*zip(*transitions))
                batch_state = nd.array(batch.state, opt.ctx).astype('float32')
                batch_state_next = nd.array(batch.next_state, opt.ctx).astype('float32')
                batch_reward = nd.array(batch.reward, opt.ctx)
                batch_action = nd.array(batch.action, opt.ctx).astype('uint8')
                batch_done = nd.array(batch.done, opt.ctx)
                target_dqn_hidden = detach(target_dqn_hidden)
                dqn_hidden = detach(dqn_hidden)
                with autograd.record():
                    outputs, _ = target_dqn(batch_state_next, target_dqn_hidden)
                    Q_sp = nd.max(outputs, axis=1)
                    Q_sp = Q_sp * (nd.ones(opt.batch_size, ctx=opt.ctx) - batch_done)
                    Q_s_array = dqn(batch_state, dqn_hidden)[0]
                    Q_s = nd.pick(Q_s_array, batch_action, 1)
                    loss = nd.mean(l2loss(Q_s, (batch_reward + opt.gamma * Q_sp)))
                loss.backward()
                DQN_trainer.step(opt.batch_size)

        t += 1
        frame_counter += 1
        if frame_counter % 100 == 0:
            print('frame_counter: ', frame_counter)

        # Save the model and update Target model
        if frame_counter > opt.replay_start_size:
            if frame_counter % opt.target_update == 0:
                check_point = frame_counter / (opt.target_update * 100)
                fdqn = './target_%s_%d' % ('dqn', int(check_point))
                dqn.save_params(fdqn)
                target_dqn.load_params(fdqn, opt.ctx)
        if done:
            if epis_count % 10. == 0.:
                results = 'epis[%d],eps[%f],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d' \
                          % (epis_count, eps, t + 1, frame_counter, cum_clipped_reward,
                             cum_reward, moving_average_clipped, moving_average)
                print(results)
    epis_count += 1
    tot_clipped_reward[int(epis_count) - 1] = cum_clipped_reward
    tot_reward[int(epis_count) - 1] = cum_reward
    if epis_count > 50.:
        moving_average_clipped = np.mean(tot_clipped_reward[int(epis_count) - 1 - 50:int(epis_count) - 1])
        moving_average = np.mean(tot_reward[int(epis_count) - 1 - 50:int(epis_count) - 1])


outfile = TemporaryFile()
outfile_clip = TemporaryFile()
np.save(outfile, moving_average)
np.save(outfile_clip, moving_average_clipped)

bandwidth = 1000  # Moving average bandwidth
total_clipped = np.zeros(int(epis_count) - bandwidth)
total_rew = np.zeros(int(epis_count) - bandwidth)
for i in range(int(epis_count) - bandwidth):
    total_clipped[i] = np.sum(tot_clipped_reward[i:i + bandwidth]) / bandwidth
    total_rew[i] = np.sum(tot_reward[i:i + bandwidth]) / bandwidth
t = np.arange(int(epis_count) - bandwidth)
belplt = plt.plot(t, total_rew[0:int(epis_count) - bandwidth], "r", label="Return")
plt.legend()  # handles[likplt,belplt])
print('Running after %d number of episodes' % epis_count)
plt.xlabel("Number of episode")
plt.ylabel("Average Reward per episode")
plt.show()
likplt = plt.plot(t, total_clipped[0:opt.num_episode - bandwidth], "b", label="Clipped Return")
plt.legend()  # handles[likplt,belplt])
plt.xlabel("Number of episode")
plt.ylabel("Average clipped Reward per episode")
plt.show()

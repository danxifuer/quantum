import tensorflow as tf
import numpy as np
import random
from collections import deque
import pandas as pd

tf.reset_default_graph()
day_len = 15
RB = pd.read_csv('/home/daiab/machine_disk/code/quantum/database/RB_day.csv', index_col=0)
RB.index = pd.DatetimeIndex(RB.index)
RB['CLOSE'] = RB['C']
RB[['O', 'H', 'L', 'C', 'V']] = (RB[['O', 'H', 'L', 'C', 'V']] - RB[['O', 'H', 'L', 'C', 'V']].rolling(
    window=5).mean()) / RB[['O', 'H', 'L', 'C', 'V']].rolling(window=5).std()
stock_data = RB.dropna()
print('Length of data:', len(stock_data))


def prepare_data(data, n_step=day_len):
    """
    data should be pd.DataFrame()
    """
    X = []
    for i in range(len(data) - n_step):
        X.append(data.iloc[i:i + n_step])
    return X


my_train = prepare_data(stock_data)


class TWStock:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_index = 0

    def render(self):
        return

    def reset(self):
        self.stock_index = 0
        return (self.stock_data[self.stock_index][['C', '3d_COLOR', '5d_COLOR', 'pos']]).as_matrix()

    # 0: 觀望, 1: 持有多單, 2: 持有空單
    def step(self, action):
        self.stock_index += 1
        if action == 0:
            action_reward = 0.5 * self.stock_data[self.stock_index - 1]['pos'][-1] * (
                self.stock_data[self.stock_index]['CLOSE'][day_len - 1] -
                self.stock_data[self.stock_index]['CLOSE'][day_len - 2])
        elif action == 1:
            action_reward = 1.0 * self.stock_data[self.stock_index - 1]['pos'][-1] * (
                self.stock_data[self.stock_index]['CLOSE'][day_len - 1] -
                self.stock_data[self.stock_index]['CLOSE'][day_len - 2])
        elif action == 2:
            action_reward = 1.5 * self.stock_data[self.stock_index - 1]['pos'][-1] * (
                self.stock_data[self.stock_index]['CLOSE'][day_len - 1] -
                self.stock_data[self.stock_index]['CLOSE'][day_len - 2])
        else:
            raise Exception('action type error')

        stock_done = False
        if self.stock_index >= len(self.stock_data) - 1:
            stock_done = True
        return (self.stock_data[self.stock_index][['C', '3d_COLOR', '5d_COLOR', 'pos']]).as_matrix(), \
               action_reward,\
               stock_done, \
               0


# Hyper Parameters for DQN
GAMMA = 0.7  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 128  # size of minibatch


def weight_variable(shape, nominate):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial, name=nominate)


def bias_variable(shape, nominate):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=nominate)


def conv1d(x, W, s=1):
    return tf.nn.conv1d(value=x, filters=W, stride=s, padding='SAME', data_format='NHWC')


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        self.state_dim = day_len
        self.action_dim = 3

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        self.W1 = weight_variable([4, 10], 'w1')
        self.b1 = bias_variable([10], 'b1')

        self.W2 = weight_variable([10, 3], 'w2')
        self.b2 = bias_variable([3], 'b2')

        self.state_input = tf.placeholder("float", [None, day_len, 4], name='input')
        self.input = tf.reshape(self.state_input, [-1, 4])

        self.input_rnn = tf.matmul(self.input, self.W1) + self.b1
        self.rnn_input = tf.reshape(self.input_rnn, [-1, day_len, 10])
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0)

        self.rnn_output, self.final_states = tf.nn.dynamic_rnn(self.lstm_cell, self.rnn_input, dtype=tf.float32)

        self.output_rnn = tf.reshape(self.final_states[-1], [-1, 10])
        self.Q_value = tf.add(tf.matmul(self.output_rnn, self.W2), self.b2, name='output')

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.98).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0]
        self.epsilon = self.epsilon - (self.epsilon - FINAL_EPSILON) / 50000
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0])


# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 10000  # Episode limitation
STEP = 1500  # 300 # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)
    env = TWStock(my_train)
    agent = DQN(env)

    print('START')
    for episode in range(EPISODE):

        # initialize task
        state = env.reset()

        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for trai
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 20 == 0:
            total_reward = 0

            for i in range(TEST):
                state = env.reset()
                env.stock_index = 1500
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward, 'Epsilon:', agent.epsilon)
            if ave_reward >= 3000:
                saver = tf.train.Saver()
                # saver=tf.train.Saver([agent.W_conv1,agent.b_conv1,agent.W_conv2,agent.b_conv2,
                # agent.W_fc1,agent.b_fc1,agent.W_fc2,agent.b_fc2])
                saver.save(agent.session, './model')
                #        saver.save(agent.session,os.path.join(os.getcwd(), 'model'))
                print('END')
                break


if __name__ == '__main__':
    main()

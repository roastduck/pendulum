#!/usr/bin/env python3

import gym
import tensorflow as tf
import shutil
import collections
import random

# Game
ENV_NAME = "Pendulum-v0"

# NN
DESCENT_RATE = 0.0001
HID_DIMS = [20]
INIT_HID_COE = [(0, 1)] # Each tuple = (mean, stddev), same below
INIT_HID_BIAS = [(0.02, 0)]
INIT_ACTOUT_COE = (0, 1)
INIT_ACTOUT_BIAS = (0.02, 0)
INIT_VOUT_COE = (0, 1)
INIT_VOUT_BIAS = (0.02, 0)
INIT_L = (0, 5)

# RL
INIT_STDDEV = 0.67
FINAL_STDDEV = 0.01
NET_VERSION_DIFF = 300
GAMMA = 0.9
EPISODE = 10000
STEP = 200
REPLAY_SIZE = 10000
BATCH_SIZE = 32
REWARD_ADDITION = 10

# Test
TEST_INTERVAL = 100
TEST_CASES = 3

# Summary
SUMMARY_INTERVAL = 1000

class Net:
    ''' Neural Network '''

    @classmethod
    def buildGraph(cls, envDim, actDim):
        ''' All nets share one graph
            Call this before construct objects
            @param envDim : dimension of environment input layer
            @param actDim : dimension of action output layer '''

        layerBefore = cls.envIn = tf.placeholder(tf.float32, (None, envDim), name = 'envIn')
        for i in range(len(HID_DIMS)):
            layerBefore = cls._genLayer('layer' + str(i), layerBefore, HID_DIMS[i], INIT_HID_COE[i], INIT_HID_BIAS[i], True)
        cls.actOut = tf.nn.sigmoid(cls._genLayer('actOut', layerBefore, actDim, INIT_ACTOUT_COE, INIT_ACTOUT_BIAS, False)) # in (0, 1)
        cls.VOut = tf.reduce_sum(cls._genLayer('VOut', layerBefore, 1, INIT_VOUT_COE, INIT_VOUT_BIAS, False), 1)
        del layerBefore

        cls.actIn = tf.placeholder(tf.float32, (None, actDim), name = 'actIn')

        with tf.name_scope('alpha'):
            alpha = cls.actIn - cls.actOut

        with tf.name_scope('L'):
            def genLElement(i, j):
                if j > i:
                    return tf.constant(0, tf.float32)
                var = tf.Variable(tf.truncated_normal((), INIT_L[0], INIT_L[1]))
                if i == j:
                    var = tf.nn.softplus(var)
                return var
            L = [[genLElement(i, j) for j in range(actDim)] for i in range(actDim)]
            tf.summary.histogram('histogram', L)
            del genLElement

        with tf.name_scope('P'):
            P = tf.matmul(L, tf.transpose(L))
            tf.summary.histogram('histogram', P)

        with tf.name_scope('A'):
            A = tf.reduce_sum(tf.matmul(alpha, P) * alpha, 1) # The last multiplication is not matmul
            tf.summary.histogram('histogram', A)

        with tf.name_scope('Q'):
            cls.QOut = -A + cls.VOut
            tf.summary.histogram('histogram', cls.QOut)

        with tf.name_scope('loss'):
            cls.QIn = tf.placeholder(tf.float32, (None), name = 'QIn')
            diffs = cls.QOut - cls.QIn
            loss = tf.reduce_mean(tf.square(diffs))
            cls.optimizer = tf.train.AdamOptimizer(DESCENT_RATE).minimize(loss)

            tf.summary.histogram('diffs', diffs)
            tf.summary.scalar('loss', loss)

        cls.globalInit = tf.global_variables_initializer()

    def __init__(self):
        ''' Constructor '''

        self.sess = tf.Session()
        self.sess.run(self.globalInit)
        self.trainCnt = 0

    def getAction(self, envTensors):
        ''' Return actions of batch of environment tensors '''

        return self.sess.run(self.actOut, { self.envIn: envTensors })

    def getV(self, envTensors):
        ''' Return Q value of batch of environment tensors '''

        return self.sess.run(self.VOut, { self.envIn: envTensors })

    # @profile
    def feed(self, sampledState, sampledAction, sampledQ):
        ''' Train '''

        self._setupLogs()
        feedDict = {
            self.envIn: sampledState,
            self.actIn: sampledAction,
            self.QIn: sampledQ
        }
        if self.trainCnt % SUMMARY_INTERVAL == 0:
            summary, _ = self.sess.run((self.summaries, self.optimizer), feedDict)
            self.summaryWriter.add_summary(summary, self.trainCnt)
        else:
            self.sess.run(self.optimizer, feedDict)
        self.trainCnt += 1

    def copyTo(self, rhs):
        ''' Copy all the values of variables to another net
            @param rhs : another net '''

        for var in tf.global_variables():
            var.load(var.eval(session = self.sess), rhs.sess)

    @classmethod
    def _genLayer(cls, name, layer0, dim, coeInit, biasInit, nonLinear):
        ''' Generate a layer with dimension `dim` and from layer `layer0`
            @param coeInit : Initial distribution for coe
            @param biasInit : Initial distribution for bias
            @param nonLinear : boolean. Whether to use nonLinear functions '''

        with tf.name_scope(name):
            coe = tf.Variable(tf.truncated_normal((layer0.get_shape().as_list()[1], dim), coeInit[0], coeInit[1]), name = 'coe')
            bias = tf.Variable(tf.truncated_normal((dim, ), biasInit[0], biasInit[1]), name = 'bias')
            res = tf.matmul(layer0, coe) + bias
            if nonLinear:
                res = tf.nn.elu(res)
            
            tf.summary.histogram('coeHistogram', coe)
            tf.summary.histogram('biasHistogram', bias)
            tf.summary.histogram('resHistogram', res)

            return res

    def _setupLogs(self):
        if hasattr(self, 'logSet'):
            return
        self.logSet = True
        shutil.rmtree('logs', True)
        self.summaryWriter = tf.summary.FileWriter('logs', self.sess.graph)
        self.summaries = tf.summary.merge_all()

class Agent:

    def __init__(self, env):
        self.env = env
        Net.buildGraph(env.observation_space.shape[0], 2)
        self.netNow = Net()
        self.netBefore = Net()
        self.replay = collections.deque()

    def reset(self):
        ''' Reset to the initial state '''

        self.state = self.env.reset()

    def optAction(self):
        ''' Return optimized action from current state '''

        return self.netNow.getAction((self.state, ))[0]

    def exploreExploitAction(self, completeness):
        ''' Return exploration-exploitation balanced action
            @param completeness : current episode / total episode number '''

        opt = self.optAction()
        disturb = lambda x: min(1.0, max(0.0, random.normalvariate(x, INIT_STDDEV + (FINAL_STDDEV - INIT_STDDEV) * completeness)))
        ret = list(map(disturb, opt))
        return ret

    # @profile
    def perceive(self, oldState, action, newState, reward, done):
        ''' Train '''

        if self.netBefore.trainCnt < self.netNow.trainCnt - NET_VERSION_DIFF:
            self.netNow.copyTo(self.netBefore) # Not updating continuously to stay stative
            self.netBefore.trainCnt = self.netNow.trainCnt
        self.replay.append({
            'state': oldState,
            'action': action,
            'Q': reward if done else reward + GAMMA * self.netBefore.getV((newState, ))
        })
        if len(self.replay) > REPLAY_SIZE:
            self.replay.popleft()
        batch = random.sample(self.replay, min(len(self.replay), BATCH_SIZE))
        self.netNow.feed([item['state'] for item in batch], [item['action'] for item in batch], [item['Q'] for item in batch])

    # @profile
    def doAction(self, action, perceive):
        ''' Perform action
            @param action : action vector (absolute impluse, probability to use positive impluse)
            @param perceive : boolean. Whether to perceive after performing action
            @return : (Reward gained, Whether done) '''
        
        newState, reward, done, _ = self.env.step((2.0 * action[0] * (1 if random.random() < action[1] else -1), ))
        if perceive:
            self.perceive(self.state, action, newState, REWARD_ADDITION + reward, done)
        self.state = newState
        return (reward, done)

# @profile
def mainLoop(env, agent):
    ''' Main loop
        Left `env` and `agent` as parameters to make debugging easier '''

    for i in range(EPISODE + 1): # +1 to enable the last test
        # print(i)
        agent.reset()
        for _ in range(STEP):
            _, done = agent.doAction(agent.exploreExploitAction(float(i) / EPISODE), True)
            if done:
                break

        # Tests
        if i % TEST_INTERVAL == 0:
            totReward = 0
            for _ in range(TEST_CASES):
                agent.reset()
                for _ in range(STEP):
                    # env.render()
                    reward, done = agent.doAction(agent.optAction(), False)
                    totReward += reward
                    if done:
                        break
            avgReward = totReward / TEST_CASES
            print("In %d-th episode, avgReward = %f"%(i, avgReward))

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    agent = Agent(env)
    mainLoop(env, agent)


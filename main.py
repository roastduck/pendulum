#!/usr/bin/env python3

import gym
import tensorflow as tf
import shutil
import collections
import random
import numpy

# Game
ENV_NAME = "Pendulum-v0"

# NN
DESCENT_RATE = 0.0001
HID_DIMS = [20, 20]
INIT_HID_COE = [(0, 1), (0, 1)] # Each tuple = (mean, stddev), same below
INIT_HID_BIAS = [(0.02, 0), (0.02, 0)]
INIT_ACTOUT_COE = (0, 1)
INIT_ACTOUT_BIAS = (0.02, 0)
INIT_VOUT_COE = (0, 1)
INIT_VOUT_BIAS = (0.02, 0)
INIT_L = (0, 1)

# RL
INIT_STDDEV = 0.67
FINAL_STDDEV = 0.01
NET_VERSION_DIFF = 300
GAMMA = 0.9
EPISODE = 10000
STEP = 300
REPLAY_SIZE = 10000
BATCH_SIZE = 32

# Test
TEST_INTERVAL = 100
TEST_CASES = 3

# Summary
SUMMARY_INTERVAL = 100

class Net:
    ''' Neural Network '''

    def __init__(self, envDim, actDim):
        ''' Constructor
            @param envDim : dimension of environment input layer
            @param actDim : dimension of action output layer '''

        layerBefore = self.envIn = tf.placeholder(tf.float32, (None, envDim))
        for i in range(len(HID_DIMS)):
            layerBefore = self._genLayer('layer' + str(i), layerBefore, HID_DIMS[i], INIT_HID_COE[i], INIT_HID_BIAS[i], True)
        self.actOut = self._genLayer('actOut', layerBefore, actDim, INIT_ACTOUT_COE, INIT_ACTOUT_BIAS, False)
        self.VOut = tf.reduce_sum(self._genLayer('VOut', layerBefore, 1, INIT_VOUT_COE, INIT_VOUT_BIAS, False), 1)
        del layerBefore

        self.actIn = tf.placeholder(tf.float32, (None, actDim))

        with tf.name_scope('alpha'):
            alpha = self.actIn - self.actOut

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
            self.QOut = -A + self.VOut
            tf.summary.histogram('histogram', self.QOut)

        with tf.name_scope('loss'):
            self.QIn = tf.placeholder(tf.float32, (None))
            diffs = self.QOut - self.QIn
            loss = tf.reduce_mean(tf.square(diffs))
            self.optimizer = tf.train.AdamOptimizer(DESCENT_RATE).minimize(loss)

            tf.summary.histogram('diffs', diffs)
            tf.summary.scalar('loss', loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.trainCnt = 0

    def getAction(self, envTensors):
        ''' Return actions of batch of environment tensors '''

        return self.sess.run(self.actOut, { self.envIn: envTensors })

    def getV(self, envTensors):
        ''' Return Q value of batch of environment tensors '''

        return self.sess.run(self.VOut, { self.envIn: envTensors })

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
        self.netNow = Net(env.observation_space.shape[0], env.action_space.shape[0])
        self.netBefore = Net(env.observation_space.shape[0], env.action_space.shape[0])
        self.replay = collections.deque()

    def reset(self):
        ''' Reset to the initial state '''

        self.state = self.env.reset()

    def optAction(self):
        ''' Return optimized action from current state '''
        # TODO: 返回的真的会在范围内吗？只有一维真的够吗？或许可以不以这个做直接输出

        return self.netNow.getAction((self.state, ))[0]

    def exploreExploitAction(self, completeness):
        ''' Return exploration-exploitation balanced action
            @param completeness : current episode / total episode number '''

        opt = self.optAction()
        disturb = lambda x: random.normalvariate(x, INIT_STDDEV + (FINAL_STDDEV - INIT_STDDEV) * completeness)
        while True:
            ret = numpy.array(list(map(disturb, opt)))
            if self.env.action_space.contains(ret):
                return ret

    def perceive(self, oldState, action, newState, reward, done):
        ''' Train '''

        if self.netBefore.trainCnt < self.netNow.trainCnt - NET_VERSION_DIFF:
            self.netNow.copyTo(self.netBefore) # Not updating continuously to stay stative
        self.replay.append({
            'state': oldState,
            'action': action,
            'Q': reward if done else reward + GAMMA * self.netBefore.getV((newState, ))
        })
        if len(self.replay) > REPLAY_SIZE:
            self.replay.popleft()
        batch = random.sample(self.replay, min(len(self.replay), BATCH_SIZE))
        self.netNow.feed([item['state'] for item in batch], [item['action'] for item in batch], [item['Q'] for item in batch])

    def doAction(self, action, perceive):
        ''' Perform action
            @param perceive : boolean. Whether to perceive after performing action
            @return : (Reward gained, Whether done) '''
        
        newState, reward, done, _ = self.env.step(action)
        if perceive:
            self.perceive(self.state, action, newState, reward, done)
        self.state = newState
        return (reward, done)

def mainLoop(env, agent):
    ''' Main loop
        Left `env` and `agent` as parameters to make debugging easier '''

    for i in range(EPISODE + 1): # +1 to enable the last test
        agent.reset()
        for j in range(STEP):
            _, done = agent.doAction(agent.exploreExploitAction(float(i) / EPISODE), True)
            if done:
                break

        # Tests
        if i % TEST_INTERVAL == 0:
            totReward = 0
            for _ in range(TEST_CASES):
                agent.reset()
                success = True
                for j in range(STEP):
                    # env.render()
                    reward, done = agent.doAction(agent.optAction(), False)
                    totReward += reward
                    if done:
                        success = False
                        print("  Failed in %s steps"%(j))
                        break
                if success:
                    print("  Success")
            avgReward = totReward / TEST_CASES
            print("In %d-th episode, avgReward = %f"%(i, avgReward))

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    agent = Agent(env)
    mainLoop(env, agent)


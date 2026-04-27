import imp
import numpy as np
from utilities_tf import load_batch_gcnn

def stan(data):
    # mu = np.mean(data, axis=0)
    # sigma = np.std(data, axis=0)

    max_ = np.max(abs(data), axis=0)

    # sigma[np.where(sigma == 0)] = 1
    max_[np.where(max_ < 1e-6)] = 1

    return data/max_


class RingBuffer(object):
    def __init__(self, maxlen, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = []  # 保存数据，使用时需要转为numpy array

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.length]  # 避免超出index

    def get_batch(self, idxs):        
        return np.array(self.data)[(self.start + idxs) % self.length]  # 转为numpy array后

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1

        self.data.append(v)
        self.data = self.data[-self.maxlen:]


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, reward_length):
        self.limit = limit

        self.states = RingBuffer(limit)  # static features
        self.dynamic0 = RingBuffer(limit)  # dynamic from features
        self.dynamic1 = RingBuffer(limit)  # dynamic to features
        self.actions = RingBuffer(limit)  # action
        self.rewards = RingBuffer(limit)  # reward

        self.actions_next = RingBuffer(limit)  # next action

        self.instance = RingBuffer(limit)  # instance
        self.var_num = RingBuffer(limit)  # variable num

        self.reward_length = reward_length

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.choice(self.nb_entries, batch_size, replace=False)

        batch_states = self.states.get_batch(batch_idxs)
        
        v_features = np.concatenate(batch_states, axis=0)

        # Static features normalization
        # v_features = stan(v_features)

        batch_dynamic0 = self.dynamic0.get_batch(batch_idxs)
        batch_dynamic1 = self.dynamic1.get_batch(batch_idxs)

        new_dynamic0 = []
        for i in range(len(batch_dynamic0)):
            new_dynamic0.append(batch_dynamic0[i].reshape(4, -1))
        
        new_dynamic1 = []
        for i in range(len(batch_dynamic1)):
            new_dynamic1.append(batch_dynamic1[i].reshape(4, -1))

        batch_dynamic0 = np.concatenate(new_dynamic0, axis=1)
        batch_dynamic1 = np.concatenate(new_dynamic1, axis=1)
        

        # batch_dynamic0 = np.concatenate(self.dynamic0.get_batch(batch_idxs), axis=1)
        # batch_dynamic1 = np.concatenate(self.dynamic1.get_batch(batch_idxs), axis=1)

        v_features0 = np.concatenate((v_features, batch_dynamic0.transpose(1,0)), axis=1)
        v_features1 = np.concatenate((v_features, batch_dynamic1.transpose(1,0)), axis=1)


        batch_action = np.concatenate(self.actions.get_batch(batch_idxs), axis=0)
        # batch_rewards = np.mean(self.rewards.get_batch(batch_idxs))
        batch_rewards = self.rewards.get_batch(batch_idxs)[:self.reward_length]

        batch_next_action = np.concatenate(self.actions_next.get_batch(batch_idxs), axis=0)

        instance_batch = self.instance.get_batch(batch_idxs)

        batch_variable_num = self.var_num.get_batch(batch_idxs)

        result = {
            'variable_features0': array_min2d(v_features0),
            'variable_features1': array_min2d(v_features1),
            'rewards': array_min2d(batch_rewards),
            'actions': array_min2d(batch_action),
            'next_actions': array_min2d(batch_next_action),
            'ins_ind': array_min2d(instance_batch),
            'variable_num': batch_variable_num
            }
        return result

    def append(self, states, action, reward, dynamic_feature0, dynamic_feature1, next_action, ins, variable_num, training=True):
        if not training:
            return

        self.states.append(states)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dynamic0.append(dynamic_feature0.flatten())
        self.dynamic1.append(dynamic_feature1.flatten())
        self.actions_next.append(next_action)

        self.var_num.append(variable_num)

        self.instance.append(ins)

    @property
    def nb_entries(self):
        return len(self.states)

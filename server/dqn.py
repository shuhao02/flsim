import logging
import random
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from server import Server
from threading import Thread
from utils.kcenter import GreedyKCenter  # pylint: disable=no-name-in-module

# 超参数
BATCH_SIZE = 32
LR = 0.1                   # learning rate
EPSILON = 0.8               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 1   # Q 现实网络的更新频率
MEMORY_CAPACITY = 10      # 记忆库大小
N_ACTIONS = 100   # 能做的动作(即能选的设备)
N_STATES = 43108000

# #阶乘
# def factorial_(n):
#     result = 1
#     for i in range(2,n+1):
#         result=result*i
#     return result

# def comb(n,m):
#     return factorial_(n)//(factorial_(n-m)*factorial_(m))              #使用自己的阶乘函数计算组合数

def trans_torch(x):
    x = torch.FloatTensor(x)
    return x

class DQN(object):
    def __init__(self):
        # 建立 target net 和 eval net 还有 memory
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            print(actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]    # return the argmax
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
        print("做出选择，选择:",action)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        print("dqn网络更新")
       # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class DQNServer(Server):
    """Federated learning server that performs dqn profiling during selection."""

    def __init__(self, config):
        self.config = config
        
    # Run federated learning
    def run(self):
        # Perform profiling on all clients
        self.dqn = DQN()
        self.profiling()

        # Designate space for storing used client profiles
        self.used_profiles = []

        # Continue federated learning
        super().run()

    def round(self):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_client_index = self.selection()
        sample_client = [client for client, _ in [self.global_weights[sample_client_index]]]

        #获取当前状态
        s = trans_torch([weight for _, weight in self.global_weights]).reshape(-1)

        # Configure sample clients
        self.configuration(sample_client)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_client]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # with Pool() as pool:
        #     processes = [pool.apply_async(client.run, ()) \
        #         for client in sample_clients]
        #     proc_results = [proc.get() for proc in processes]

        # Recieve client updates
        reports = self.reporting(sample_client)
        # reports = self.reporting(sample_clients, proc_results)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        #更新report_weight到global_weight,(转updated_weight为一维)
        self.global_weights[sample_client_index] = (sample_client[0],self.flatten_weights(reports[0].weights))

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)
        
        #dqn环境反馈和训练
        #需要获得s,a,r,s_（disabled）,状态为所有client的weight
        s_ = trans_torch([weight for _, weight in self.global_weights]).reshape(-1)
        print("action:",sample_client_index)

        #这里的2可以修改，作为一个参数
        r = (pow(2,(accuracy-self.config.fl.target_accuracy)) - 1 ) * 10
        print("reward:",r)
        self.dqn.store_transition(s, sample_client_index, r, s_)
        if self.dqn.memory_counter > MEMORY_CAPACITY:
            self.dqn.learn() # 记忆库满了就进行学习
        s = s_


        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy
    
    # Federated learning phases
    def selection(self):
        # Select devices to participate in round

        # profiles = self.profiles
        # k = self.config.clients.per_round

        # if len(profiles) == 0 :  # Reuse clients when needed
        #     logging.warning('Not enough unused clients')
        #     logging.warning('Dumping clients for reuse')
        #     self.profiles.extend(self.used_profiles)
        #     self.used_profiles = []

        # Shuffle profiles 打乱客户端
        # random.shuffle(profiles)

        # weights like [[weight1(tensor)],[weight2],...]
        weights = trans_torch([weight for _, weight in self.global_weights]).reshape(-1)
        
        #每次通过dqn选择client来进行更新权重，action返回动作,例如0，1，2，3...，
        sample_client_index = self.dqn.choose_action(weights)
        

        # Mark sample profiles as used 它使用了一个不断减少可选择客户端的方法来使得客户端的选择不陷入极端
        # self.used_profiles.extend(sample_profiles)
        # for i in sorted(centers_index, reverse=True):
        #     del self.profiles[i]
        
        return sample_client_index


    def profiling(self):

        # Use all clients for profiling
        clients = self.clients

        # Configure clients for training
        self.configuration(clients)

        # Train on clients to generate profile weights
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports = self.reporting(clients)

        # Extract weights from reports
        weights = [report.weights for report in reports]
        weights = [self.flatten_weights(weight) for weight in weights]

        #设置全局权重，作为选择时的依据
        self.global_weights = [(client, weights[i])
                         for i, client in enumerate(clients)]

        # Use weights for client profiles
        self.profiles = [(client, weights[i])
                         for i, client in enumerate(clients)]
        return self.profiles

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
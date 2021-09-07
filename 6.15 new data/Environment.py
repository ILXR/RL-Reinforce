import pandas as pd
import numpy as np
import random


class Environment():
    def __init__(self):
        super().__init__()
        tem = pd.read_csv("./6.15 new data/[2021.06.10]DNN_Partition.csv")
        all_delay = self.partition_data = tem[tem['epoch']==1]
        all_flop = self.flop_data = pd.read_csv("./6.15 new data/[2021.06.10]cuflopFlag.csv")
        self.actions = np.arange(0, 23)
        max_delay,min_delay = all_delay['allDelay'].max(),all_delay['allDelay'].min()
        max_flop,min_flop = all_flop['S_flops'].max(),all_flop['S_flops'].min()
        all_data = pd.DataFrame(columns=('gpu','batch','cut','reward'))
        count = 0
        for item in all_delay.iterrows():
            [epoch,gpu,batch,cut,delay,_,_] = item[1].values
            delay_reward = (delay-min_delay)/(max_delay-min_delay)
            power = all_flop[(all_flop['batchNum']==batch) & (all_flop['modelCutID_T']==cut)]['S_flops'].values[0]
            power_reward = (power-min_flop)/(max_flop-min_flop)
            all_data.loc[count] = {'gpu':gpu,'batch':batch,'cut':cut,'reward':delay_reward+power_reward}
            count+=1
        self.data_frame = all_data
        self.__count = count    
        
    def get_rewards(self, state):
        [gpu,batch] = state
        data = self.data_frame
        res = data[(data['gpu']==int(gpu))&(data['batch']==int(batch))]['reward'].values
        return np.array(res)

    def get_reward(self, state, action):
        ''' state = [gpu,batch] '''
        [gpu,batch] = state
        data = self.data_frame
        res = data[(data['gpu']==int(gpu))&(data['batch']==int(batch))&(data['cut']==int(action))]['reward'].values[0]
        return res
    
    def get_reward_by_action(self,action):
        data = self.data_frame
        reward = data[(data['cut']==action)]['reward']
        index = random.randint(0, reward.shape[0]-1)
        return reward.iloc[index]
        # return reward.max()

    def best_step(self, state):
        ''' return -> best_cut, best_reward'''
        [gpu,batch] = state
        data = self.data_frame
        tem_data = data[(data['gpu']==int(gpu))&(data['batch']==int(batch))]
        best_reward = tem_data['reward'].min()
        best_cut = data[data['reward']==best_reward]['cut'].values[0]
        return best_cut, best_reward

    def random_step(self):
        ''' return -> [gpu,batch],cut,reward '''
        index = random.randint(0,self.__count-1)
        data = self.data_frame.loc[index]
        return np.array([data['gpu'],data['batch']]), data['cut'], data['reward']

    def random_state(self):
        ''' return -> [gpu,batch] '''
        index = random.randint(0,self.__count-1)
        data = self.data_frame.iloc[index]
        return np.array([data['gpu'],data['batch']],dtype='double')

    def action_size(self):
        return len(self.actions)

    def state_size(self):
        return 2

    def action_space(self):
        return self.actions

    def data_size(self):
        return self.__count

# env = Environment()
# print(env.random_state())
# [gpu,batch],cut,reward = env.random_step()
# print(gpu,batch,cut,reward)
# print(env.get_reward([gpu,batch],cut))
# print(env.get_reward_by_action(cut))
# print(env.best_step([gpu,batch]))
# print(env.action_size())
# print(env.action_space())
# print(env.state_size())
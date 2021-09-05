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
            power = all_flop[(all_flop['batchNum']==batch) & (all_flop['modelCutID_T']==cut)]['S_flops'].mean()
            power_reward = (power-min_flop)/(max_flop-min_flop)
            all_data.loc[count] = {'gpu':gpu,'batch':batch,'cut':cut,'reward':delay_reward+power_reward}
            count+=1
        self.all_data = all_data
    

    def get_reward(self, state):
        ''' state = [gpu,batch] '''
        [gpu,batch] = state
        data = self.data_frame
        res = data[(data['gpu']==gpu)&(data['batch']==batch)]['reward'].values[0]
        return res
    
    def get_reward_by_action(self,action):
        data = self.all_data
        reward = data[(data['cut']==action)]['reward'].mean()
        return reward

    def best_step(self, state):
        
        return -1, 0

    def random_step(self):
        index = random.randint(0, self.__state_size)
        data = self.data_frame.loc[index]
        return [int(data['gpu']),int(data['cut'])], data['reward']

    def random_state(self):
        index = random.randint(0, self.__state_size)
        data = self.data_frame.loc[index]
        return [data['gpu'],data['cut']]

    def action_size(self):
        return len(self.actions)

    def state_size(self):
        return self.__state_size

    def action_space(self):
        return self.actions

env = Environment()
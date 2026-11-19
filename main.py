from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard 
from keras.optimizers import Adam 
import numpy as np 
from collections import deque 
import time 


REPLAY_MEMORY_SIZE = 50_000 # ez 50,000 
MODEL_NAME = "256x2" #--> 256 yby 2 cog net

#what are batches why matte i frogot
#how ca nit take inm sooo much at once lol 
#what are hyperparameters
#what is an episode 

class DQNA:

    def __init__(self):
        #main model --> gets trained every step 
        self.model = self.create_model()
        #target model --> this is what we .predict against every step --> LEARN 
        self.targetModel = self.create_model()
        self.targetModel.set_weights(self.model.get_weights)

        self.replay_memory = deque(maxLen = REPLAY_MEMORY_SIZE) #array that has max length

        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256,(3,3), input_shape = env.OBSERVATIONS_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE), Activation = "linear")
        model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
        return model 

def update_replay_memory(self,transition):
    self.replay_memory.append(transition) #observation space, reward, new observation space, flag ( those values ) 

def get_qs(self,state, step):
    return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0] # why divide by 255? 
        
    #leaky relu prevents dying relu 
     
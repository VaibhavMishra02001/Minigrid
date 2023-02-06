import gym
import numpy as np 

import random
import minigrid
from minigrid import Window

env = gym.make('MiniGrid-Empty-5x5-v0')
window = Window("minigrid")
total_episodes = 300
learning_rate = 0.1
gamma = 0.96
actions=[0,1,2]
state_space=[]
Q=[]
state_space=[0,0,0]
Q =[[state_space,0]]
i=1
def greedy(x,y,direction1):
    action=0
   
    if np.random.uniform(0, 1) < epsilon:
         action = random.choice(actions)
                       
    else:
        for i in range (len(Q)):
            if(state_space[i]==[x,y,direction1]):
                Q[i][1]=actions
                action= np.argmax(Q[i])
                break
        
                
       
    return action
step=1
def sarsa(x1,y1,direction1,x2,y2,direction2, reward, action,action2):
   for i in range (len(Q)):
        if(state_space[i]==[x1,y1,direction1]):
            Q[i][1]=action
            prediction=Q[i]
            for j in range (len(state_space)):
                if(state_space[j]==[x2,y2,direction2]):
                    Q[j][1]=action2
                    target = reward + gamma * Q[j]
                    Q[j]=Q[j]+learning_rate(prediction-target)



epsilon=1
decay=1.07
for episode in range(total_episodes):
    epsilon=epsilon/decay
    env.reset()
    x1,y1=env.agent_pos
    direction1= env.agent_dir
    state=[x1,y1,direction1]
    if state not in state_space:
        state_space.append(state)
    
    action1=greedy(x1,y1,direction1)
    if  [state,action1] not in Q:
        Q.append([state,action1])
    else:
        for i in range (len(Q)):
            if(Q[i]==[state,action1]):
                Q[i][1]=action1

   
    terminated=False
    while not terminated:
   
        obs,reward,terminated,info,_=env.step(action1)
        x2,y2=env.agent_pos
        direction2=env.agent_dir
        state2=[x2,y2,direction2]
        
        action2=greedy(x2,y2,direction2)
        
        if state2 not in state_space:
            state_space.append(state)
        if  [state2,action2] not in Q:
            Q.append([state2,action2])
        else:
            for i in range (len(Q)):
                if(Q[i]==[state2,action2]):
                    Q[i][1]=action2
       
        
        sarsa(x1, y1,direction1,x2,y2,direction2, reward, action1,action2)
        x1=x2
        y1=y2
        direction1=direction2
        action1=action2
        img = env.get_frame()
        window.show_img(img)  
    print("Episode:",step)  
    step=step+1   
env.close()
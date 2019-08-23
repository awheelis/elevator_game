import numpy as np
import matplotlib.pyplot as plt
from os import system, name 
import random
import time

def clear(): 

    if name == 'nt': 
        _ = system('cls') 
    
    else: 
        _ = system('clear') 



# ELEVATOR:

# Capacity:
# 12

# | E |  876567
# |   |  5434
# |   |  543234
# |   |  98778
# |   |  312321234312
# |   |  76
# |   |  4444446643
# |   |  43223

# Reward:
# -1 for every time step
# +1 for every delivery
# This is an elevator simulation. 

# People on floors are randomly generated and their destinations are randomly generated

EMPTY_SPACE = 99 

class ElevatorEnv(object):

    def __init__(self, elevators = 1, floors = 8, people_on_each_floor = 5):
        self.floors = floors
        self.people_on_each_floor = people_on_each_floor
        self.elevators = elevators
        self.elevator_capacity = 3
        self.blank_state = ['_' for i in range(self.elevator_capacity)]

        self.ele_map = [['_' for i in range(self.elevator_capacity)] for i in range(self.floors)]
        self.floor_map = [['_' for i in range(self.people_on_each_floor)] for i in range(self.floors)]

        self.stateSpace = [i for i in range(self.floors * (self.elevator_capacity + people_on_each_floor))]   # this also doesn't make any sense
        self.stateSpacePlus  = [i for i in range(self.floors * (self.elevator_capacity + people_on_each_floor))]  # this doesn't make any sense
        self.agentPosition = 0
        self.loc_elevator = 0
        self.doorstate = 0
        self.peopleInEle = []
        
        self.actionSpace = {'U': -self.loc_elevator, 'D': self.loc_elevator,'C': 0, 'O': 1}
        self.possibleActions = ['U', 'D', 'O','C']
        self.setFloors()

    def getElevatorInfo(self):
        floor = self.loc_elevator
        door = self.doorstate
        peopleInEle = self.peopleInEle
        return door, floor, peopleInEle

    def elevatorFunction(self,  floor):  # HELP
        ele_map = self.ele_map[floor]
        for person in ele_map:
            if person == self.loc_elevator:
                ele_map.remove(person)
                ele_map.append('_')
        for person in ele_map:
            if person == '_':
                try:
                    ele_map.remove(person)
                    ele_map.append(self.floor_map[floor].pop())
                except:
                     ele_map.append('_')
        self.ele_map[floor] = ele_map
        

    def isTerminalState(self):
        count = 0
        for floor in self.floor_map:
            for person in floor:
                count += 1
        if count == 0:
            return True
        else:
            return False

    def setState(self, agentDoor, agentFloor, agentPeople): # this is supposed to move the people after the elevator goes up and down
        doorstate, loc_elevator, peopleInEle = getElevatorInfo()
        self.doorstate = 0
        self.ele_map[loc_elevator] = np.zeros(self.elevator_capacity)
        self.peopleInEle = []
        agentDoor, agentFloor, agentPeople = state
        self.doorstate = agentDoor
        self.ele_map[self.loc_elevator] = agentPeople
        self.peopleInEle = agentPeople
        self.agentPosition = self.doorstate + self.loc_elevator + len(self.peopleInEle)

    def moveEle(self, newLoc):
        if self.doorstate == 0:
            peopleInEle = self.ele_map[self.loc_elevator]
            self.ele_map[self.loc_elevator] = self.blank_state
            self.ele_map[newLoc] = peopleInEle
            self.loc_elevator = newLoc


    def setFloors(self):
        floor_list = [i for i in range(self.floors)]
        self.primary_floor = floor_list[random.randint(0, len(floor_list)- 1)]
        self.secondary_floor = floor_list[random.randint(0, len(floor_list)- 2)]
        self.third_floor = floor_list[random.randint(0, len(floor_list)- 3)]

        self.total_people = self.floors * self.people_on_each_floor
       
        for floor in range(self.floors):
            for person in range(self.people_on_each_floor):
                rand = int(str(random.randint(0, 10))[-1])
                if rand >= 0 and rand < 6:
                    self.floor_map[floor][person] = self.primary_floor
                elif rand <9 and rand > 5:
                    self.floor_map[floor][person] = self.secondary_floor
                else:
                    self.floor_map[floor][person] = self.third_floor

    def step(self, action):
        
        if action == "O" or action == "C":
            self.doorstate = self.actionSpace[action]
            if self.doorstate == 1:
                self.peopleInEle = self.elevatorFunction(self.loc_elevator) 
            reward = -1

        else:
            if action == 'U':  
                new_ele_loc = max(self.loc_elevator - 1, 0)
                self.moveEle(new_ele_loc)
            else:
                new_ele_loc = min(self.loc_elevator + 1, self.floors - 1)
                self.moveEle(new_ele_loc)

            reward = -6

        return self.agentPosition, reward, self.isTerminalState(), None

    def render(self):
        print('----------------------')
        for major_row in range(self.floors):
            print(major_row)
            if major_row == self.loc_elevator:
                print(">")
            else:
                print(" ")
            for ele_per in self.ele_map[major_row]:
                print(ele_per, end = '\t')
            print('|', end = '')
            for floor_per in range(len(self.floor_map[major_row])):
                print(self.floor_map[major_row][floor_per], end = '\t')
            print('\n')
        print('----------------------')
        
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def reset(self):
        self.ele_map = [['_' for i in range(self.elevator_capacity)] for i in range(self.floors)]
        self.floor_map = [['_' for i in range(self.people_on_each_floor)] for i in range(self.floors)]
        self.loc_elevator = 0
        self.doorstate = 0
        self.setFloors()
        return self.loc_elevator

def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

def getRandomAction():
    actions = ['U', 'C', 'O', 'D']
    return random.choice(actions)

if __name__ == '__main__':
    # model hyperparameters
    env = ElevatorEnv(elevators = 1, floors = 8, people_on_each_floor = 7)
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    env.render()
    clear()
    for i in range(numGames):
        if i % 1 == 0:

            print('starting game ', i)
            time.sleep(1)
        moveList = []
        done = False
        epRewards = 0
        observation = env.reset()
        moveNumber = 0
        while not done:
            env.render()
            time.sleep(.25)
            rand = np.random.random()
            action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) else env.actionSpaceSample()
            moveList.append(action)
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + GAMMA*Q[observation_,action_] - Q[observation,action])
            observation = observation_
            clear()
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()

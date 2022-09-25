#!/usr/bin/env python3
import argparse
from collections import deque
from functools import total_ordering
import numpy as np
import pathlib
from lib import environ, models,wrapper

import torch
import csv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SAVES_DIR = pathlib.Path("saves")

STATE_STEPS = 1
PASSENGER_DATA_DIR = "data/line_2/passenger"
BUS_DATA_DIR = "data/line_2/bus"
MANUAL_DISPATCH_FILE = "data/line_2/bus/direction_0_dispatch.csv"
MAX_EPISODE_STEPS = 1050
STATS_AGGREGATE = 1
# min and max headway betweeb subsequent dispatch actions.
MIN_HEADWAY = 3
MAX_HEADWAY = 10

def load_manual_dispatch(path):
    dispatches = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            dispatches += [int(row[0])]
    return dispatches

#init result dictionary
status = {
        "minute":deque(),
        "boarded": deque(),
        "alighted": deque(),
        "left": deque(),
        "stranded": deque(),
        "waiting_minutes":deque(),
        "running_buses":deque()
    }

measurements = {
    "minute": deque(),
    "action":deque(),
    "reward": deque(),
    "waiting_time":deque(),
    "boarded":deque(),
    "alighted":deque(),
    "left":deque(),
    "stranded":deque(),
    "invalid_dispatch":deque(),
    "headway":deque(),
    "invalid_headway":deque(),
    "running_buses":deque()
}
# dispatch every 5 minutes during peak hours (7:00-10:00 and 16:00-19:00), every 10 minutes otherwise
def uniform_action(time, obs_v):
    if(time%10 == 0):
        return environ.Actions.DISPATCH
    else:
        return environ.Actions.HOLD

# dispatch every 5 minutes during peak hours (7:00-10:00 and 16:00-19:00), every 10 minutes otherwise
def static_action(time, obs_v):
    if(time%5 != 0):
        return environ.Actions.HOLD
    elif (time >= 420 and time <= 600 or time >= 960 and time <=1140):
        return environ.Actions.DISPATCH
    elif time%10 == 0:
        return environ.Actions.DISPATCH
    else:
        return environ.Actions.HOLD

# original manual dispatch
def manual_action(time, obs_v):
    if(time%5 != 0):
        return environ.Actions.HOLD
    elif (time >= 420 and time <= 600 or time >= 960 and time <=1140):
        return environ.Actions.DISPATCH
    elif time%10 == 0:
        return environ.Actions.DISPATCH
    else:
        return environ.Actions.HOLD

tmp = {
   "acc_boarded" : 0, 
   "acc_alighted" : 0,
   "acc_left":0
}
def step_record(minute, env, action, is_invalid_dispatch, reward):
        (list_last_boarded, list_last_alighted,list_last_left,  
         list_last_stranded, list_last_waiting_minutes, list_running_buses)  = env.get_last_step_stats()
        
        status["minute"].append(minute)
        status["boarded"].append(list_last_boarded)
        status["alighted"].append(list_last_alighted)
        status["left"].append(list_last_left)
        status["stranded"].append(list_last_stranded)
        status["waiting_minutes"].append(list_last_waiting_minutes)
        status["running_buses"].append(list_running_buses)

        tmp["acc_boarded"]+= np.sum(list_last_boarded)
        tmp["acc_alighted"]+= np.sum(list_last_alighted)
        tmp["acc_left"]+= np.sum(list_last_left)
        #print(tmp)
        #step metrics:
        action = 1 if(action == environ.Actions.DISPATCH) else 0
        measurements["minute"].append(minute)
        measurements["reward"].append(reward)
        measurements["invalid_dispatch"].append(1 if is_invalid_dispatch else 0)
        measurements["waiting_time"].append(np.sum(list_last_waiting_minutes))
        measurements["boarded"].append(np.sum(list_last_boarded))
        measurements["alighted"].append(np.sum(list_last_alighted))
        measurements["stranded"].append(np.sum(list_last_stranded))
        measurements["left"].append(np.sum(list_last_left))
        measurements["running_buses"].append(np.sum(list_running_buses))
        last_dispatch = np.argwhere(np.array(measurements["action"])==1).reshape(-1)
        last_dispatch = max(last_dispatch) if len(last_dispatch)>0 else 0
        headway = step_idx - last_dispatch
        measurements["headway"].append(headway)
        measurements["invalid_headway"].append(1 if (action == 1 and ((headway>0 and headway < MIN_HEADWAY) or headway > MAX_HEADWAY)) else 0)
        measurements["action"].append(action)

def save_result(path, name):
    n_stations = len(status["boarded"][0])
    detail_data = None
    detail_headers = []
    n_steps = len(measurements["minute"])
    for key in status:
        all_stations = np.array(status[key]).reshape(n_steps,-1)
        if key == "minute":
                detail_headers += [key]
        else:
                detail_headers += [f"{key}_{i}" for i in range(n_stations)]
        if detail_data is None:
                detail_data = all_stations
        else:
                detail_data = np.concatenate((detail_data, all_stations),axis=1)

    np.savetxt(f"{path}/{name}_detail_status.csv", detail_data, delimiter=',', fmt='%d', header=','.join(detail_headers), comments='')
    df = pd.DataFrame.from_dict(measurements)
    df.to_csv(f"{path}/{name}_step_metrics.csv")


def calc_reward(riders, left, tot_stranded, dispatches, tot_waiting_time):
    return riders*2 + left *(-5) + dispatches*(-30) + tot_waiting_time*(-0.2) + tot_stranded*(-2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("-n", "--name", required=False, default="manual", help="Model name to be used.(manual/)")
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    args = parser.parse_args()
    
    passenger_dir = pathlib.Path(PASSENGER_DATA_DIR).absolute()
    bus_dir = pathlib.Path(BUS_DATA_DIR).absolute()
       
    env = environ.BusEnv(state_steps=1, passenger_dir=passenger_dir, bus_dir=bus_dir)

    manual_dispatches = []
    if args.name == "manual":
        manual_dispatches = load_manual_dispatch(MANUAL_DISPATCH_FILE)

    obs = env.reset()
    
    net = None #static policy
    if args.name == "att": #attention
        net = models.TransformDQN(obs.shape, env.action_space.n)
    elif args.name == "cnn": #cnn
        net = models.DQNConvNStepBusBased(obs.shape, env.action_space.n)
    elif args.name == "fc": #fully connect
        net = models.simpleFF(obs.shape, env.action_space.n)
    if net != None:
        net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    time = env.get_time()

    #print("initial state:", obs)
    _, rows, cols = env.observation_space.shape
    total_reward = 0.0
    step_idx = 0
    rewards = []
    all_actions = np.zeros(int(MAX_EPISODE_STEPS/STATS_AGGREGATE))
    stats = np.zeros([int(MAX_EPISODE_STEPS/STATS_AGGREGATE), rows, cols])
    total_runs = 0
    total_alighted = 0
    total_strended = 0
    total_left = 0
    invalid_dispatch = deque()
    while True:
        obs_v = torch.tensor([obs])
        if args.name == "uniform":
            action_idx = uniform_action(time, obs_v)
        elif args.name == "schedule":
            action_idx = static_action(time, obs_v)
        elif args.name == "manual":
            action_idx = environ.Actions.DISPATCH if time in manual_dispatches else environ.Actions.HOLD
        else:
            out_v = net(obs_v)
            action_idx = out_v.max(dim=1)[1].item()
        action = environ.Actions(action_idx)
        
        if action == environ.Actions.DISPATCH:
            total_runs += 1
            print(f"time{time}, step{step_idx}============run a bus===========")
            all_actions[step_idx] = 10
        obs, reward, done, info = env.step(action_idx)

        if info["invalid_dispatch"]:
            print(f"invalid dispatch, no bus available at time {time}")
            invalid_dispatch.append(step_idx)
        total_reward += reward
        rewards.append(total_reward)
        # print(env.get_trip_loads(True))
        if step_idx % 100 == 0:
            print("%d: reward=%.3f" % (step_idx, total_reward))
        
        step_record(time, env, action, info["invalid_dispatch"], reward)
        
        step_idx += 1
        time += 1
        if done:
            break
    print("=======================summary===================================")
    print(f"total reward from model:{total_reward}, invalid dispatchs:{invalid_dispatch}")
    #print_stats(env)
    print(tmp)
    print(f"total dispatch:{sum(measurements['action'])}")
    save_result(SAVES_DIR, args.name)

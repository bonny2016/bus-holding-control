#!/usr/bin/env python3
from collections import deque
from os import sync
import ptan
import pathlib
import argparse
import gym.wrappers
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import environ, models, common, validation
import pandas as pd

SAVES_DIR = pathlib.Path("saves")
PASSENGER_DATA_DIR = "data/line_2/passenger"
BUS_DATA_DIR = "data/line_2/bus"

VALIDATION_PASSENGER_DATA_DIR = "data/line_2/passenger"
VALIDATION_BUS_DATA_DIR = "data/line_2/bus"

BATCH_SIZE = 64

EPS_START = 1.0
EPS_FINAL = 0.01
EPS_STEPS = 10000
GAMMA = 0.99

REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000
REWARD_STEPS = 1
LEARNING_RATE = 0.001
STATES_TO_EVALUATE = 1000
STAGING_STEPS = 0
MAX_EPISODE_STEPS = 1100
STATE_STEPS = 1

def sync_model(target, src):
    target.load_state_dict(src.state_dict())

def save_result(results, path):
    df = pd.DataFrame(data=results)
    df.to_csv(f"{path}/result.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=False, default="fc", help="DQN network architecture(fc/cnn/att)")
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    parser.add_argument("-s", "--steps", required=False, default=STATE_STEPS, help="n previous steps")
    parser.add_argument("-r", "--run", required=False, default="", help="Run name")
    args = parser.parse_args()
    if args.steps:
      STATE_STEPS = int(args.steps)

    passenger_dir = pathlib.Path(PASSENGER_DATA_DIR).absolute()
    bus_dir = pathlib.Path(BUS_DATA_DIR).absolute()

    val_passenger_dir = pathlib.Path(VALIDATION_PASSENGER_DATA_DIR).absolute()
    val_bus_dir = pathlib.Path(VALIDATION_BUS_DATA_DIR).absolute()

    env = environ.BusEnv(state_steps=STATE_STEPS, passenger_dir=passenger_dir, bus_dir=bus_dir)
    
    obs = env.reset()


    env_val = environ.BusEnv(state_steps=STATE_STEPS, passenger_dir=val_passenger_dir, bus_dir=val_bus_dir)
    env_val.reset()

    path_pattern = f"state_{STATE_STEPS}_staging_{STAGING_STEPS}_episode_{MAX_EPISODE_STEPS}_{args.run}"
    
    if args.name == "att": #attention
        net = models.TransformDQN(obs.shape, env.action_space.n)
        tgt_net = models.TransformDQN(obs.shape, env.action_space.n)
        path_pattern = f"att-{path_pattern}"
    elif args.name == "fc": #fully connect
        net = models.simpleFF(obs.shape, env.action_space.n)
        tgt_net = models.simpleFF(obs.shape, env.action_space.n)
        path_pattern = f"fc-{path_pattern}"
    elif args.name == "cnn": #cnn
        net = models.DQNConvNStepBusBased(obs.shape, env.action_space.n)
        tgt_net = models.DQNConvNStepBusBased(obs.shape, env.action_space.n)
        path_pattern = f"cnn-{path_pattern}"
    else:
        print("invalid name argument. please use one of these: (att, fc, cnn)")
        exit(0)
    saves_path = SAVES_DIR / path_pattern
    saves_path.mkdir(parents=True, exist_ok=True)

    if args.model is not None:
        net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage)) 
        tgt_net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage)) 
    
    print("model:")
    print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = ptan.actions.EpsilonTracker(
        selector, EPS_START, EPS_FINAL, EPS_STEPS)
    
    agent = ptan.agent.DQNAgent(net, selector)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    deq_rewards = deque()
    deq_iter = deque()
    results = {
        "val_iter" : deq_iter,
        "val_rewards":deq_rewards,
    }
    
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss(
            batch, net, tgt_net,
            gamma=GAMMA ** REWARD_STEPS)
        #print(f"loss:{loss_v:.2f}")
        loss_v.backward()
        optimizer.step()
        eps_tracker.frame(engine.state.iteration)
        
        if getattr(engine.state, "eval_states", None) is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.array(transition.state, copy=False)
                           for transition in eval_states]
            engine.state.eval_states = np.array(eval_states, copy=False)
        
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
            "lr": optimizer.param_groups[0]["lr"]
        }
        
    engine = Engine(process_batch)
    tb = common.setup_ignite(engine, exp_source, f"{path_pattern}",
                             extra_metrics=('values_mean',))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)
    def sync_eval(engine: Engine):
        #tgt_net.sync()
        sync_model(tgt_net, net)
        mean_val = common.calc_values_of_states(
            engine.state.eval_states, net)
        engine.state.metrics["values_mean"] = mean_val
        print(f"mean_val:{mean_val:.2f}, epsilon:{selector.epsilon}, lr:{optimizer.param_groups[0]['lr']}")
        if getattr(engine.state, "best_mean_val", None) is None:
            engine.state.best_mean_val = mean_val
        if engine.state.best_mean_val < mean_val:
            print("%d: Best mean value updated %.3f -> %.3f" % (
                engine.state.iteration, engine.state.best_mean_val,
                mean_val))
            path = saves_path / ("mean_value-%.3f.data" % mean_val)
            torch.save(net.state_dict(), path)
            engine.state.best_mean_val = mean_val

    @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def validate(engine: Engine):
        res = validation.validation_run(env_val, net, episodes=1, epsilon=0)
        print("%d: val: %s" % (engine.state.iteration, res))
        deq_rewards.append(res["episode_reward"])
        deq_iter.append(engine.state.iteration)
        for key, val in res.items():
            engine.state.metrics[key + "_val"] = val
        val_reward = res['episode_reward']
        save_result(results, path = saves_path)
        if getattr(engine.state, "best_val_reward", None) is None:
            engine.state.best_val_reward = val_reward
        if engine.state.best_val_reward < val_reward:
            print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                engine.state.best_val_reward, val_reward
            ))
            engine.state.best_val_reward = val_reward
            path = saves_path / ("val_reward-%.3f.data" % val_reward)
            torch.save(net.state_dict(), path)
    event = ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED
    
    val_metrics = [m + "_val" for m in validation.METRICS]
    val_handler = tb_logger.OutputHandler(
        tag="validation", metric_names=val_metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)
    
    # train 300k ierations
    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE),max_epochs=30, epoch_length=REPLAY_SIZE)
    df = pd.DataFrame(data=results)
    df.to_csv("result.csv",index=False)
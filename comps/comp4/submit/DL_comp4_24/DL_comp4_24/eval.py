import os
import ctypes
import copy
import tensorflow as tf
import numpy as np
import pandas as pd
import models.A2C as A2C
import envs.cartPole as cartPole
import envs.flappyBird as flappyBird
import models.util as Util

def init_agent():
    env = flappyBird.FlappyBirdEnv()
    NUM_STATE_FEATURES = env.get_num_state_features()
    NUM_ACTIONS = env.get_num_actions()
    LEARNING_RATE = 0.0001
    REWARD_DISCOUNT = 0.99
    COEF_VALUE= 1
    COEF_ENTROPY = 0
    agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, COEF_VALUE, COEF_ENTROPY)

    return agent

def recover(agent):
    # Setup recorder
    ckpt = tf.train.Checkpoint(model=agent.model, opt=agent.optimizer)
    recorder = Util.Recorder(ckpt=ckpt, ckpt_path='results/ckpt', plot_title='A3C FlappyBird', filename='results/a3c_flappy', save_period=5000)
    ep = recorder.restore()
    return agent

def TA_state():
    state = copy.deepcopy(game.getGameState())
    
    state['next_next_pipe_bottom_y'] -= state['player_y']
    state['next_next_pipe_top_y'] -= state['player_y']
    state['next_pipe_bottom_y'] -= state['player_y']
    state['next_pipe_top_y'] -= state['player_y']
    relative_state = list(state.values())


    # return the state in tensor type, with batch dimension
    relative_state = tf.convert_to_tensor(relative_state, dtype=tf.float32)
    relative_state = tf.expand_dims(relative_state, axis=0)
    
    return relative_state


if __name__ == '__main__':
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line disable pop-out window
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    agent = init_agent()
    # Setup recorder
    ckpt = tf.train.Checkpoint(model=agent.model, opt=agent.optimizer)
    recorder = Util.Recorder(ckpt=ckpt, ckpt_path='results/ckpt', plot_title='A3C FlappyBird', filename='results/a3c_flappy', save_period=5000)
    ep = recorder.restore()
    model = agent.model
    model_path = 'saved/eval/flappy_eval'
    model.save(model_path)
    model_path = 'DL_comp4_24_model'
    model.save(model_path)

    print(f"Model Saved at {model_path}")

    #TA　code
    # set it True if your model returns multiple values
    multiple_return_values = True

    # set visible GPU
    gpu_number = 3

    # print out more information
    verbose = False

    # public seed is 2021
    seed = 2021


    os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line disable pop-out window
    from ple.games.flappybird import FlappyBird
    from ple import PLE

    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False, rng=seed)  # game environment interface
    env.reset_game()

    alive_times = []
    episode_rewards = []

    for test_num in range(1, 101):
        alive_time = 1
        episode_reward = 0
        env.reset_game()

        print(f"Epoch {test_num}")
        while not env.game_over():
            state = TA_state()

            # Your model should return action probabilities
            # In other words, the last layer of your model should be Softmax
            
            if not multiple_return_values:
                action_prob = model(state)
            else:
                action_prob = model(state)[0]
            
            if verbose:
                print(f"test num: {test_num}, frame: {alive_time}, action probs: {action_prob}")
                
            action_idx = tf.argmax(action_prob, axis=1)[0]

            reward = env.act(env.getActionSet()[action_idx])

            alive_time += 1
            episode_reward += reward
            
        alive_times.append(alive_time)
        episode_rewards.append(episode_reward)

        if verbose:
            print(f"[{test_num}] alive: {alive_time}, episode reward: {episode_reward}")
            
        print(f"[{test_num}] alive: {alive_time}, episode reward: {episode_reward}")
        
    print(f"average alive time: {np.mean(np.asarray(alive_times))},\naverage episode reward: {np.mean(np.asarray(episode_rewards))}\nshow your result https://docs.google.com/spreadsheets/d/1QHNmes31XdUSsG2K9U7cgTggeGfiMgADvrJJsETjbxM")
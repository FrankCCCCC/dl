from multiprocessing import Process, Lock, Value, Array, Queue, cpu_count, Event
import os
import ctypes
import tensorflow as tf
import numpy as np
import pandas as pd
import models.A2C as A2C
import envs.cartPole as cartPole
import envs.flappyBird as flappyBird
import models.util as Util

class A3C:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        tf.config.set_soft_device_placement(True)

    def worker(self, proc_id, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queue, global_res_queue, event, is_load):
        print(f'Process {proc_id} Worker {worker_id} start')

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])

        # Set Process Priority
        # Util.highpriority()

        with tf.device("/CPU:0"):
            local_agent, local_env, n_step = self.init_agent_env(proc_id, 'worker', worker_id)

            # Reset the weight back to checkpoint
            ckpt = tf.train.Checkpoint(model=local_agent.model, opt=local_agent.optimizer)
            recorder = Util.Recorder(ckpt=ckpt, ckpt_path='results/ckpt', plot_title='A3C FlappyBird', filename='results/a3c_flappy', save_period=500)

            # Restore from checkpoint
            ep = 0
            if is_load:
                ep = recorder.restore()
                print(f"Worker {worker_id} Restore {ep}")
            else:
                print(f"Worker {worker_id} No Restore")

            # After recovering, wait for main process wake up to continue
            event.wait()

            # Copy model from the global agent
            global_vars = global_var_queue.get()
            local_agent.model.set_weights(global_vars)

            # Reset Game State
            state = local_env.reset()

            while global_remain_episode.value > 0:
                is_over = False
                episode_reward = 0

                # Update n step
                while not is_over:
                    # Interact n steps
                    reward, loss, gradients, trajectory, is_over = local_agent.train_on_env(env = local_env, n_step = n_step, cal_gradient_vars = None)
                    episode_reward = episode_reward + reward

                    # Update
                    global_grad_queue.put({'loss': loss, 'reward': episode_reward, 'gradients': gradients, 'is_over': is_over, 'worker_id': worker_id})
                    if not global_var_queue.empty():
                        global_vars = global_var_queue.get()
                        local_agent.model.set_weights(global_vars)
                        # local_agent.model.set_weights(global_vars['model'])
                        # local_agent.optimizer.set_weights(global_vars['opt'])
                        # print(f'Worker {worker_id} Update Weights')

                # print(f'Episode {global_remain_episode.value} Reward with worker {worker_id}: {episode_reward}')
                global_res_queue.put({'loss': loss, 'reward': episode_reward, 'worker_id': worker_id})
                with global_remain_episode.get_lock():
                    global_remain_episode.value -= 1

        with global_alive_workers.get_lock():
            global_alive_workers.value -= 1

        print(f"Worker {worker_id} done")

    def param_server(self, proc_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues, is_load):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # Set Process Priority
        # Util.highpriority()

        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        with tf.device("/CPU:0"):
            global_agent, env, n_step = self.init_agent_env(proc_id, 'ps', ps_id)

            # Setup recorder
            ckpt = tf.train.Checkpoint(model=global_agent.model, opt=global_agent.optimizer)
            recorder = Util.Recorder(ckpt=ckpt, ckpt_path='results/ckpt', plot_title='A3C FlappyBird', filename='results/a3c_flappy', save_period=500)
            
            # Restore from checkpoint
            ep = 0
            if is_load:
                ep = recorder.restore()
                print(f"Master Restore {ep}")
            else:
                print(f"Master No Restore")

            with global_remain_episode.get_lock():
                global_remain_episode.value = global_remain_episode.value - ep

            # Copy model to the local agent
            model_weights = global_agent.model.get_weights()
            for i in range(len(global_var_queues)):
                global_var_queues[i].put(model_weights)

            while ((not global_grad_queue.empty()) or (global_alive_workers.value > 0)):
                if not global_grad_queue.empty():
                    # print(f'Getting gradients from queue')
                    item = global_grad_queue.get()
                    global_agent.update(loss = item['loss'], gradients = item['gradients'])

                    # Record when an episode is over
                    if item['is_over']:
                        recorder.record(float(item['loss']), float(item['reward']))

                    model_weights = global_agent.model.get_weights()
                    # opt_weights = global_agent.optimizer.get_weights()
                    for i in range(global_alive_workers.value):
                        if not global_var_queues[i].full():
                            global_var_queues[i].put(model_weights)
                            # global_var_queues[i].put({'model': model_weights, 'opt': opt_weights})
                            # print(f'Put vars in queue for worker {i}')
            
            print("Complete PS apply")
            for queue in global_var_queues:
                if not queue.empty():
                    queue.get()
                    # print(f'Clear vars in queue for worker')
            print(f'PS {ps_id} done')

    def init_agent_env(self, proc_id, role, role_id):
        # env = cartPole.CartPoleEnv()
        env = flappyBird.FlappyBirdEnv()
        NUM_STATE_FEATURES = env.get_num_state_features()
        NUM_ACTIONS = env.get_num_actions()
        LEARNING_RATE = 0.0001
        REWARD_DISCOUNT = 0.99
        COEF_VALUE= 1
        COEF_ENTROPY = 0
        n_step = None
        agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, COEF_VALUE, COEF_ENTROPY)

        return agent, env, n_step

    # def is_having_training_info(self):
    #     return ((not global_res_queue.empty()) or (global_alive_workers.value > 0))
    def get_res(self, global_res_queue, global_alive_workers):
        if ((not global_res_queue.empty()) or (global_alive_workers.value > 0)):
            return global_res_queue.get()
        else:
            return None

    def start(self):
        # print(tf.config.experimental.list_physical_devices(device_type=None))
        # print(tf.config.experimental.list_logical_devices(device_type=None))
        epoc = 3000000
        worker_num = 10
        max_workers = 20
        update_freq_ep = 100
        is_load = True

        self.episode_num = epoc
        self.ps_num = 1
        self.base_worker_num = worker_num
        self.worker_num = worker_num 
        self.max_workers = max_workers 
        self.target_workers = worker_num 
        self.avg_ep_time = 0
        self.avg_ep_time_w = 0.04
        self.last_update_ep = 1
        self.current_episode = 1
        self.update_freq_ep = update_freq_ep

        global_remain_episode = Value('i', self.episode_num)
        global_alive_workers = Value('i', self.worker_num)
        global_res_queue = Queue()
        global_grad_queue = Queue()
        global_var_queues = [Queue(1) for i in range(self.max_workers)]
        events = [Event() for i in range(self.max_workers)]

        pss = []
        workers = []
        episode_results = []
        
        for ps_id in range(self.ps_num):
            pss.append(Process(target = self.param_server, args=(ps_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues, is_load)))

        for worker_id in range(self.max_workers):
            workers.append(Process(target = self.worker, args=(worker_id + self.ps_num, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues[worker_id], global_res_queue, events[worker_id], is_load)))

        for num in range(self.ps_num):
            pss[num].start()

        for num in range(self.max_workers):
            workers[num].start()

        for num in range(self.worker_num):
            events[num].set()

        while ((not global_res_queue.empty()) or (global_alive_workers.value > 0)):
            if not global_res_queue.empty():
                episode_results.append(global_res_queue.get())
                episode_res = episode_results.pop(0)
                print(f"Episode {self.current_episode} Reward with worker {episode_res['worker_id']}: {episode_res['reward']}\t| Loss: {episode_res['loss']}")
                self.current_episode += 1

                if self.avg_ep_time > 20:
                    self.target_workers = int(self.avg_ep_time // 10 + self.base_worker_num)
                    if self.target_workers > self.max_workers:
                        self.target_workers = self.max_workers
                    # print(f"Target Worker: {self.target_workers} | Worker Num {self.worker_num}")
                    # print(f"current_episode: {self.current_episode} - last_update_ep: {self.last_update_ep} > update_freq_ep: {self.update_freq_ep}")

                if (self.target_workers > self.worker_num) and (self.current_episode - self.last_update_ep > self.update_freq_ep):
                    # print(f"Adding New Worker")
                    events[self.worker_num].set()
                    with global_alive_workers.get_lock():
                        global_alive_workers.value += 1

                    self.worker_num += 1
                    self.last_update_ep = self.current_episode

                    print(f"Add New Worker!!! | Total Worker {self.worker_num}")
            
        global_grad_queue.close()
        global_grad_queue.join_thread()

        global_res_queue.close()
        global_res_queue.join_thread()

        for queue in global_var_queues:
            queue.close()
            queue.join_thread()

        for num in range(self.worker_num):
            workers[num].join()
            print(f'Worker {num} join')

        for num in range(self.ps_num):
            pss[num].join()
            print(f'PS {num} join')

if __name__ == '__main__':
    # print(tf.config.experimental.list_physical_devices(device_type=None))
    # print(tf.config.experimental.list_logical_devices(device_type=None))

    a3c = A3C()
    a3c.start()
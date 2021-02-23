import tensorflow as tf
import numpy as np

# ======================================

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ======================================

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line make pop-out window not appear
from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)  # environment interface to game
env.reset_game()

# ======================================

# Define Input Size
IMG_WIDTH = 84
IMG_HEIGHT = 84
NUM_STACK = 4
# For Epsilon-greedy
MIN_EXPLORING_RATE = 0.01

# ======================================

class Agent:
    def __init__(self, name, num_action, discount_factor=0.99):
        self.exploring_rate = 0.1
        self.discount_factor = discount_factor
        self.num_action = num_action
        self.model = self.build_model(name)

    def build_model(self, name):
        # input: state
        # output: each action's Q-value 
        screen_stack = tf.keras.Input(shape=[IMG_WIDTH, IMG_HEIGHT, NUM_STACK], dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.num_action)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model
    
    def loss(self, state, action, reward, tar_Q, ternimal):
        # Q(s,a,theta) for all a, shape (batch_size, num_action)
        output = self.model(state)
        index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
        # Q(s,a,theta) for selected a, shape (batch_size, 1)
        Q = tf.gather_nd(output, index)
        
        # set tar_Q as 0 if reaching terminal state
        tar_Q *= ~np.array(terminal)

        # loss = E[r+max(Q(s',a',theta'))-Q(s,a,theta)]
        loss = tf.reduce_mean(tf.square(reward + self.discount_factor * tar_Q - Q))

        return loss
    
    def max_Q(self, state):
        # Q(s,a,theta) for all a, shape (batch_size, num_action)
        output = self.model(state)

        # max(Q(s',a',theta')), shape (batch_size, 1)
        return tf.reduce_max(output, axis=1)
    
    def select_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.num_action)  # Select a random action
        else:
            state = np.expand_dims(state, axis = 0)
            # Q(s,a,theta) for all a, shape (batch_size, num_action)
            output = self.model(state)

            # select action with highest action-value
            action = tf.argmax(output, axis=1)[0]

        return action
    
    def update_parameters(self, episode):
        self.exploring_rate = max(MIN_EXPLORING_RATE, min(0.5, 0.99**((episode) / 30)))

    def shutdown_explore(self):
        # make action selection greedy
        self.exploring_rate = 0

# ======================================

# init agent
num_action = len(env.getActionSet())

# agent for frequently updating
online_agent = Agent('online', num_action)

# agent for slow updating
target_agent = Agent('target', num_action)
# synchronize target model's weight with online model's weight
target_agent.model.set_weights(online_agent.model.get_weights())

# ======================================

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
average_loss = tf.keras.metrics.Mean(name='loss')

@tf.function
def train_step(state, action, reward, next_state, ternimal):
    # Delayed Target Network
    tar_Q = target_agent.max_Q(next_state)
    with tf.GradientTape() as tape:
        loss = online_agent.loss(state, action, reward, tar_Q, ternimal)
    gradients = tape.gradient(loss, online_agent.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, online_agent.model.trainable_variables))
    
    average_loss.update_state(loss)

# ======================================

class Replay_buffer():
    def __init__(self, buffer_size=50000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
        """
        sample experience from buffer
        """
        if size > len(self.experiences):
            experiences_idx = np.random.choice(len(self.experiences), size=size)
        else:
            experiences_idx = np.random.choice(len(self.experiences), size=size, replace=False)

        # from all sampled experiences, extract a tuple of (s,a,r,s')
        states = []
        actions = []
        rewards = []
        states_prime = []
        terminal = []
        for i in range(size):
            states.append(self.experiences[experiences_idx[i]][0])
            actions.append(self.experiences[experiences_idx[i]][1])
            rewards.append(self.experiences[experiences_idx[i]][2])
            states_prime.append(self.experiences[experiences_idx[i]][3])
            terminal.append(self.experiences[experiences_idx[i]][4])

        return states, actions, rewards, states_prime, terminal

# ======================================

# init buffer
buffer = Replay_buffer()

# ======================================

import moviepy.editor as mpy

def make_anim(images, fps=60, true_image=False):
    duration = len(images) / fps

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip

# ======================================

import skimage.transform

def preprocess_screen(screen):
    screen = skimage.transform.resize(screen, [IMG_WIDTH, IMG_HEIGHT, 1])
    return screen

def frames_to_state(input_frames):
    if(len(input_frames) == 1):
        state = np.concatenate(input_frames*4, axis=-1)
    elif(len(input_frames) == 2):
        state = np.concatenate(input_frames[0:1]*2 + input_frames[1:]*2, axis=-1)
    elif(len(input_frames) == 3):
        state = np.concatenate(input_frames + input_frames[2:], axis=-1)
    else:
        state = np.concatenate(input_frames[-4:], axis=-1)

    return state

# ======================================

from IPython.display import Image, display

update_every_iteration = 1000
print_every_episode = 500
save_video_every_episode = 5000
NUM_EPISODE = 20000
NUM_EXPLORE = 20
BATCH_SIZE = 32

iter_num = 0
for episode in range(0, NUM_EPISODE + 1):
    
    # Reset the environment
    env.reset_game()
    
    # record frame
    if episode % save_video_every_episode == 0:
        frames = [env.getScreenRGB()]
    
    # input frame
    input_frames = [preprocess_screen(env.getScreenGrayscale())]
    
    # for every 500 episodes, shutdown exploration to see the performance of greedy action
    if episode % print_every_episode == 0:
        online_agent.shutdown_explore()
    
    # cumulate reward for this episode
    cum_reward = 0
    
    t = 0
    while not env.game_over():
        
        state = frames_to_state(input_frames)
        
        # feed current state and select an action
        action = online_agent.select_action(state)
        
        # execute the action and get reward
        reward = env.act(env.getActionSet()[action])
        
        # record frame
        if episode % save_video_every_episode == 0:
            frames.append(env.getScreenRGB())
        
        # record input frame
        input_frames.append(preprocess_screen(env.getScreenGrayscale()))
        
        # cumulate reward
        cum_reward += reward
        
        # observe the result
        state_prime = frames_to_state(input_frames)  # get next state
        
        # append experience for this episode
        if episode % print_every_episode != 0:
            buffer.add((state, action, reward, state_prime, env.game_over()))
        
        # Setting up for the next iteration
        state = state_prime
        t += 1
        
        # update agent
        if episode > NUM_EXPLORE and episode % print_every_episode != 0:
            iter_num += 1
            train_states, train_actions, train_rewards, train_states_prime, terminal = buffer.sample(BATCH_SIZE)
            train_states = np.asarray(train_states).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)
            train_states_prime = np.asarray(train_states_prime).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)
            
            # convert Python object to Tensor to prevent graph re-tracing
            train_states = tf.convert_to_tensor(train_states, tf.float32)
            train_actions = tf.convert_to_tensor(train_actions, tf.int32)
            train_rewards = tf.convert_to_tensor(train_rewards, tf.float32)
            train_states_prime = tf.convert_to_tensor(train_states_prime, tf.float32)
            terminal = tf.convert_to_tensor(terminal, tf.bool)
            
            train_step(train_states, train_actions, train_rewards, train_states_prime, terminal)

        # synchronize target model's weight with online model's weight every 1000 iterations
        if iter_num % update_every_iteration == 0 and episode > NUM_EXPLORE and episode % print_every_episode != 0:
            target_agent.model.set_weights(online_agent.model.get_weights())

    # update exploring rate
    online_agent.update_parameters(episode)
    target_agent.update_parameters(episode)

    if episode % print_every_episode == 0 and episode > NUM_EXPLORE:
        print(
            "[{}] time live:{}, cumulated reward: {}, exploring rate: {}, average loss: {}".
            format(episode, t, cum_reward, online_agent.exploring_rate, average_loss.result()))
        average_loss.reset_states()

    if episode % save_video_every_episode == 0:  # for every 500 episode, record an animation
        clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
        clip.write_videofile("movie_f/DQN_demo-{}.webm".format(episode), fps=60)
#         display(clip.ipython_display(fps=60, autoplay=1, loop=1, maxduration=120))

# ======================================

from moviepy.editor import *
clip = VideoFileClip("movie_f/DQN_demo-20000.webm")
display(clip.ipython_display(fps=60, autoplay=1, loop=1, maxduration=120))
# External
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time

# Internal
from Agents.Utils import Score
from Agents.Utils import preprocess_batch

def preprocess_state(state, config):
    if config.convolutional_input == False:
        return state
    """
    Takes in raw Atari image, returns 84x84 resized/scaled grayscale image
    state: should be 210 x 160 x 3 shaped np.array
    output: 1x84x84 image
    """
    cropped = Image.fromarray(state)\
        .crop((0, 34, 160, 160 + 34))
    composite = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((96, 96))
    ])
    image = composite(cropped)
    small_img = np.uint8(image)
    return np.expand_dims(small_img, 0)


#prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class Runner():

    def __init__(self, agent, save_best_score=None, verbose=1, pbar=None):
        self.agent = agent
        self.save_best_score = save_best_score
        self.verbose = verbose
        self.pbar = pbar

    def run_agent(self):

        score = Score(target_average=self.agent.config.target_average, window_size=100, total_episodes=self.agent.config.n_episodes, verbose=self.verbose, pbar=self.pbar)

        best_checkpoint = None

        for i_episode in range(1, self.agent.config.n_episodes+1):

            # reset envirnoment
            state = preprocess_state(self.agent.env.reset(), self.agent.config)
            score.reset_episode()

            # run the episode
            for t in range(self.agent.config.max_t):

                # act
                self.agent.reset()
                action = self.agent.act(state)

                # step
                next_state, reward, done, _ = self.agent.env.step(action)
                next_state = preprocess_state(next_state, self.agent.config)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state

                # collect reward
                score.add_reward(reward)
                if done:
                    break 

            # post episode handling by agent
            self.agent.post_episode()

            # handle scoring
            found_best_score = score.post_episode(i_episode)

            # save checkpoint
            if found_best_score != None:
                best_checkpoint = self.agent.get_checkpoint()
                if self.save_best_score != None:
                    self.agent.save(self.save_best_score)

        return score, best_checkpoint

    def enjoy_checkpoint(self, checkpoint):

        self.agent.load(checkpoint)

        while True:

            # reset envirnoment
            state = self.agent.config.env.reset()

            # run the episode
            for t in range(self.agent.config.max_t):

                # render
                self.agent.config.env.render()

                # act
                action = self.agent.act_no_training(state)

                # step
                next_state, _, done, _ = self.agent.config.env.step(action)
                state = next_state

                # break if done
                if done:
                    break 

    def random(self):

        env = self.agent.config.env

        while True:

            # reset envirnoment
            state = env.reset()

            # run the episode
            for t in range(self.agent.config.max_t):

                # render
                env.render()

                # act
                action = env.action_space.sample()

                # step
                next_state, _, done, _ = env.step(action)
                state = next_state

                # break if done
                if done:
                    break 

    def enjoy_checkpoint_frame_batches_probabilities(self, checkpoint):

        self.agent.load(checkpoint)

        env = self.agent.config.env
        policy = self.agent.policy

        self.agent.config.RIGHT = 4
        self.agent.config.LEFT = 5

        while True:
        
            env.reset()

            fr1, re1, is_done, _ = env.step(0)
            fr2, re2, is_done, _ = env.step(0)

            for t in range(self.agent.config.max_t):

                # render
                env.render()
                
                # prepare the input
                # preprocess_batch properly converts two frames into 
                # shape (n, 2, 80, 80), the proper input for the policy
                # this is required when building CNN with pytorch
                batch_input = preprocess_batch(self.agent.config.device, [fr1,fr2])
                
                # probs will only be used as the pi_old
                # no gradient propagation is needed
                # so we move it to the cpu
                probs = policy(batch_input).squeeze().cpu().detach().numpy()
                
                action = np.where(np.random.rand(1) < probs, self.agent.config.RIGHT, self.agent.config.LEFT)
                probs = np.where(action==self.agent.config.RIGHT, probs, 1.0-probs)
                
                # advance the game (0=no action)
                # we take one action and skip game forward
                fr1, re1, is_done, _ = env.step(action)
                fr2, re2, is_done, _ = env.step(0)

                time.sleep(0.04)

                # stop if any of the trajectories is done
                # we want all the lists to be retangular
                if is_done:
                    break

    # collect trajectories for a parallelized parallelEnv object
    def collect_trajectories(self):
        
        envs = self.agent.envs
        policy = self.agent.policy
        tmax = self.agent.config
        nrand = 5

        # number of parallel instances
        n=len(envs.ps)

        #initialize returning lists and start the game!
        state_list=[]
        reward_list=[]
        prob_list=[]
        action_list=[]
        envs.reset()
        
        # start all parallel agents
        envs.step([1]*n)
        
        # perform nrand random steps
        for _ in range(nrand):
            fr1, re1, _, _ = envs.step(np.random.choice([self.agent.config.RIGHT, self.agent.config.LEFT],n))
            fr2, re2, _, _ = envs.step([0]*n)
        
        for t in range(self.agent.config.max_t):

            # prepare the input
            # preprocess_batch properly converts two frames into 
            # shape (n, 2, 80, 80), the proper input for the policy
            # this is required when building CNN with pytorch
            batch_input = preprocess_batch(self.agent.config.device, [fr1,fr2])
            
            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu

            probs = policy(batch_input).squeeze().cpu().detach().numpy()
            action = np.where(np.random.rand(n) < probs, self.agent.config.RIGHT, self.agent.config.LEFT)
            probs = np.where(action==self.agent.config.RIGHT, probs, 1.0-probs)

            # advance the game (0=no action)
            # we take one action and skip game forward
            fr1, re1, is_done, _ = envs.step(action)
            fr2, re2, is_done, _ = envs.step([0]*n)

            reward = re1 + re2
            
            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)
            
            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list

    def run_parallel_trajectories(self):

        score = Score(target_average=self.agent.config.target_average, window_size=100, total_episodes=self.agent.config.n_episodes, verbose=self.verbose, pbar=self.pbar)

        best_checkpoint = None

        for i_episode in range(1, self.agent.config.n_episodes+1):

            # collect trajectories
            old_probabilities, states, actions, rewards = self.collect_trajectories()
            total_rewards = np.sum(rewards, axis=0)

            # learn
            self.agent.learn(old_probabilities, states, actions, rewards)

            # handle scoring
            score.set_total_reward(np.mean(total_rewards))

            # handle scoring
            found_best_score = score.post_episode(i_episode)

            # save checkpoint
            if found_best_score != None:
                best_checkpoint = self.agent.get_checkpoint()
                if self.save_best_score != None:
                    self.agent.save(self.save_best_score)

        return score, best_checkpoint

    def run_single_probability_trajectory(self):

        score = Score(target_average=self.agent.config.target_average, window_size=100, total_episodes=self.agent.config.n_episodes, verbose=self.verbose, pbar=self.pbar)

        best_checkpoint = None

        for i_episode in range(1, self.agent.config.n_episodes+1):

            # reset envirnoment
            state = self.agent.env.reset()
            score.reset_episode()

            # run the episode
            saved_log_probabilities = []
            saved_rewards = []
            for t in range(self.agent.config.max_t):

                # log probability for the action
                action, log_probability = self.agent.act(state)
                saved_log_probabilities.append(log_probability)

                # step
                next_state, reward, done, _ = self.agent.env.step(action)
                saved_rewards.append(reward)
                state = next_state

                # collect reward
                score.add_reward(reward)
                if done:
                    break

            #  learn
            self.agent.learn(saved_log_probabilities, saved_rewards) 

            # handle scoring
            found_best_score = score.post_episode(i_episode)

            # save checkpoint
            if found_best_score != None:
                best_checkpoint = self.agent.get_checkpoint()
                if self.save_best_score != None:
                    self.agent.save(self.save_best_score)

        return score, best_checkpoint
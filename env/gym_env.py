from .simulators.game import Game
from .simulators.gridgame import GridGame
from .obs_interfaces.observation import *
import numpy as np
import json
from utils.discrete import Discrete
from utils.box import Box
import gym
# import gym_miniworld
# import gym_minigrid


class GymEnv(GridGame, GridObservation, Game, VectorObservation):
    def __init__(self, conf):
        self.game_name = conf['game_name']
        if 'MiniGrid' in self.game_name:
            colors = conf.get('colors', [(255, 255, 255), (0, 0, 0), (245, 245, 245)])
            super(GymEnv, self).__init__(conf, colors)
            self.env_core = gym.make(self.game_name)
            self.action_dim = self.env_core.action_space.n
            self.input_dimension = self.env_core.observation_space['image'].shape
            _ = self.reset()
            self.is_act_continuous = False
            self.is_obs_continuous = True

        else:
            Game.__init__(self, conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                          conf['game_name'], conf['agent_nums'], conf['obs_type'])

            self.env_core = gym.make(self.game_name)
            self.max_step = int(conf["max_step"])
            self.done = False
            self.input_dimension = self.env_core.observation_space

            self.init_info = None
            self.step_cnt = 0
            self.won = {}
            self.n_return = [0] * self.n_player

            if self.game_name != 'FetchPickAndPlace-v1' and self.game_name != 'HandManipulatePen-v0' and self.game_name != 'HandManipulateBlock-v0':
                self.load_action_space(conf)

                observation = self.env_core.reset()
                if not isinstance(observation, np.ndarray):
                    observation = np.array(observation)
                obs_list = observation.reshape(-1).tolist()

                self.joint_action_space = self.set_action_space()
                self.current_state = [obs_list] * self.n_player
                self.all_observes = self.get_all_observes()

                self.action_dim = self.get_action_dim()

                self.ob_space = [self.env_core.observation_space for _ in range(self.n_player)]
                self.ob_vector_shape = [self.env_core.observation_space.shape] * self.n_player
                self.ob_vector_range = [self.env_core.observation_space.low,
                                        self.env_core.observation_space.high] * self.n_player
            else:
                self.reach_goal = False
                self.current_state = self.env_core.reset()
                self.joint_action_space = self.set_action_space()
                self.action_dim = self.joint_action_space
                self.all_observes = self.get_all_observes()

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if self.is_act_continuous:
                if ("high" not in input_action) or ("low" not in input_action) or ("shape" not in input_action):
                    raise Exception("act_box in continuous case must have fields low, high, shape")
                shape = tuple(input_action["shape"])
                self.env_core.action_space = Box(input_action["low"], input_action["high"], shape, np.float32)
            else:
                if "discrete_n" not in input_action:
                    raise Exception("act_box in discrete case must have field discrete_n")
                discrete_n = int(input_action["discrete_n"])
                self.env_core.action_space = Discrete(discrete_n)

    def step(self, joint_action):
        if 'MiniGrid' in self.game_name:
            action = joint_action
            info_before = self.step_before_info()
            next_state, reward, self.done, info_after = self.get_next_state(action)
            self.current_state = next_state
            if isinstance(reward, np.ndarray):
                reward = reward.tolist()
            reward = self.get_reward(reward)
            self.step_cnt += 1
            done = self.is_terminal()
            self.all_observes = self.get_all_observes()
        elif self.game_name != 'FetchPickAndPlace-v1' and self.game_name != 'HandManipulatePen-v0' and self.game_name != 'HandManipulateBlock-v0':
            action = self.decode(joint_action)
            info_before = self.step_before_info()
            next_state, reward, self.done, info_after = self.get_next_state(action)
            if isinstance(reward, np.ndarray):
                reward = reward.tolist()[0]
            reward = self.get_reward(reward)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)
            next_state = next_state.reshape(-1).tolist()
            self.current_state = [next_state] * self.n_player
            self.all_observes = self.get_all_observes()
            done = self.is_terminal()
            if 'MiniWorld' in self.game_name:
                info_after = self.parse_info(info_after)
            self.step_cnt += 1
        else:
            info_before = ''
            action = self.decode(joint_action)
            observation, reward, self.done, info = self.env_core.step(action)
            if info['is_success']:
                self.reach_goal = True
                self.done = True
            self.current_state = observation
            self.all_observes = self.get_all_observes()
            reward = self.get_reward(reward)
            self.step_cnt += 1
            done = self.is_terminal()
            if done:
                self.set_final_n_return()
            info_after = info
        return self.all_observes, reward, done, info_before, info_after

    def decode(self, joint_action):

        if not self.is_act_continuous:
            return joint_action[0][0].index(1)
        else:
            return joint_action[0][0]

    def is_valid_action(self, joint_action):

        if len(joint_action) != self.n_player:
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

        for i in range(self.n_player):
            if not self.is_act_continuous:
                if len(joint_action[i][0]) != self.joint_action_space[i][0].n:
                    raise Exception("The input action dimension for player {} should be {}, not {}".format(
                        i, self.joint_action_space[i][0].n, len(joint_action[i][0])))
            else:
                if not isinstance(joint_action[i][0], np.ndarray):
                    raise Exception("For continuous action, the input of player {} should be numpy.ndarray".format(i))
                if joint_action[i][0].shape != self.joint_action_space[i][0].shape:
                    raise Exception("The input action dimension for player {} should be {}, not {}".format(
                        i, self.joint_action_space[i][0].shape, joint_action[i][0].shape))

    def get_next_state(self, action):
        if 'MiniGrid' in self.game_name:
            action = int(np.array(action[0][0]).argmax())
            observation, reward, done, info = self.env_core.step(action)
            return observation, reward, done, info
        else:
            observation, reward, done, info = self.env_core.step(action)
            if 'MiniWorld' not in self.game_name:
                obs_list = observation.tolist()
                return obs_list, reward, done, info
            else:
                return observation, reward, done, info

    def get_reward(self, reward):
        if 'MiniGrid' in self.game_name:
            return [reward]
        else:
            r = [0] * self.n_player
            for i in range(self.n_player):
                r[i] = reward
                self.n_return[i] += r[i]
            return r

    def step_before_info(self, info=''):
        return info

    def parse_info(self, info):
        new_info = {}
        for key, val in info.items():
            if isinstance(val, np.ndarray):
                new_info[key] = val.tolist()
            else:
                new_info[key] = val
        return new_info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        return self.done

    def set_final_n_return(self):
        if self.reach_goal:
            for i in range(self.n_player):
                self.n_return[i] += self.max_step
        else:
            for i in range(self.n_player):
                self.n_return[i] = 0

    def set_action_space(self):
        if 'MiniGrid' in self.game_name:
            action_space = [[Discrete(7)] for _ in range(self.n_player)]
            return action_space
        else:
            if self.is_act_continuous:
                action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
            else:
                action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
            return action_space

    def check_win(self):
        if 'MiniGrid' in self.game_name:
            return True
        elif 'MiniWorld' in self.game_name:
            if self.env_core.near(self.env_core.box):
                return 1
            else:
                return -1
        elif self.game_name == 'FetchPickAndPlace-v1' or self.game_name == 'HandManipulatePen-v0' or self.game_name == 'HandManipulateBlock-v0':
            return ''
        else:
            return '0'

    def reset(self):
        if 'MiniGrid' in self.game_name:
            obs = self.env_core.reset()
            self.step_cnt = 0
            self.done = False
            self.current_state = obs
            self.all_observes = self.get_all_observes()
            return self.all_observes
        elif self.game_name != 'FetchPickAndPlace-v1' and self.game_name != 'HandManipulatePen-v0' and self.game_name != 'HandManipulateBlock-v0':
            observation = self.env_core.reset()
            self.step_cnt = 0
            self.done = False
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation)
            obs_list = observation.reshape(-1).tolist()
            self.current_state = [obs_list] * self.n_player
            self.all_observes = self.get_all_observes()
            return self.all_observes
        else:
            self.done = False
            self.reach_goal = False
            self.current_state = self.env_core.reset()
            self.all_observes = self.get_all_observes()
            self.init_info = None
            self.step_cnt = 0
            self.won = {}
            self.n_return = [0] * self.n_player

    def get_action_dim(self):
        action_dim = 1
        print("joint action space is ", self.joint_action_space[0][0])
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space[0])):
            action_dim *= self.joint_action_space[0][i].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_vector_obs_config(self, player_id):
        return self.ob_vector_shape[player_id], self.ob_vector_range[player_id]

    def get_vector_many_obs_space(self, player_id_list):
        all_obs_space = {}
        for i in player_id_list:
            m = self.ob_vector_shape[i]
            all_obs_space[i] = m
        return all_obs_space

    def get_vector_observation(self, current_state, player_id, info_before):
        return self.current_state[player_id]

    def set_seed(self, seed=None):
        self.env_core.seed(seed)

    def get_all_observes(self):
        all_observes = []
        if 'MiniGrid' in self.game_name:
            for i in range(self.n_player):
                each = {"obs": self.current_state, "controlled_player_index": i}
                all_observes.append(each)
        elif self.game_name == 'FetchPickAndPlace-v1' or self.game_name == 'HandManipulatePen-v0' or self.game_name == 'HandManipulateBlock-v0':
            each = {"obs": self.current_state, "controlled_player_index": 0, "task_name": self.game_name}
            all_observes.append(each)
        else:
            for i in range(len(self.current_state)):
                each = {"obs": self.current_state[i], "controlled_player_index": i}
                all_observes.append(each)
        return all_observes

    def get_grid_observation(self, current_state, player_id, info_before):
        return current_state

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete

# update bounds to center around agent
cam_range = 100

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.post_step_callback = post_step_callback

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # [-1,1]
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(
                    world.dim_c,), dtype=np.float32)  # [0,1]
            
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
            agent.action.c = np.zeros(self.world.dim_c)
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()  # core.step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                d = self.world.dim_p
            else:
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    d = 5
                else:
                    if self.force_discrete_action:
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0
                    agent.action.u = action[0][0:self.world.dim_p]
                    d = self.world.dim_p

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            if (not agent.silent) and (not isinstance(action_space, MultiDiscrete)):
                action[0] = action[0][d:]
            else:
                action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False):
        from . import rendering
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            
        

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            WINDOW_W = 700
            WINDOW_H = 700
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                self.viewers[i] = rendering.Viewer(WINDOW_W, WINDOW_H)

        self.agents_geoms = []
        self.agents_geoms_xform = []
        self.render_geoms = []
        self.render_geoms_xform = []
        entities_rearrange = self.world.entities
        agents = []
        for entity in self.world.entities:
            if 'agent' in entity.name:
                agents.append(entity)
                entities_rearrange.pop(0)
        entities_rearrange += agents

        for entity in entities_rearrange:
            if 'agent' in entity.name:
                geom = rendering.make_square(entity.size, angle=entity.state.p_ang+np.pi/4)
            elif 'landmark' in entity.name:
                if entity.center:
                    geom = rendering.make_circle(entity.size)
                else:
                    geom = rendering.make_circle(entity.size)
            else:
                geom = rendering.make_circle(entity.size)
            geom.set_color(*entity.color)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

        import copy
        agents_end = agents
        agents_begin = copy.deepcopy(agents)
        agents_begin.insert(0, agents_begin.pop())
        lines = []
        lines_length_text = []
        for i in range(len(agents_begin)):
            line = rendering.Line((agents_begin[i].state.p_pos[0], agents_begin[i].state.p_pos[1]), 
                                  (agents_end[i].state.p_pos[0], agents_end[i].state.p_pos[1]))
            lines.append(line)
            agent_begin_name = agents_begin[i].name
            agent_end_name = agents_end[i].name
            tmp_text = agent_begin_name + ' to ' + agent_end_name + ' : '
            length = np.linalg.norm(agents_begin[i].state.p_pos - agents_end[i].state.p_pos)
            tmp_text = tmp_text + str(length)
            lines_length_text.append(tmp_text)


        # add geoms to viewer
        for viewer in self.viewers:
            viewer.geoms = []
            viewer.labels = []

            for geom in self.render_geoms:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from . import rendering
            # update bounds to center around agent
            #cam_range = 500
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            pos = self.world.agents[0].state.p_pos
            #self.viewers[i].set_bounds(0, 1200, 0, 1200)
            self.viewers[i].set_bounds(
                pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(entities_rearrange):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # add goal to render
            goal = []
            goal_xform = []
            goal = rendering.make_circle(5)
            goal.set_color(1.0, 0, 0)
            goal_xform = rendering.Transform()
            goal.add_attr(goal_xform)
            goal_xform.set_translation(*self.world.landmarks[0].state.p_pos)
            self.viewers[i].add_geom(goal)

            goal_ang = self.world.agents[0].state.p_ang + self.world.agents[0].ang2goal
            goal_dir = 20 * np.array([np.cos(goal_ang), np.sin(goal_ang)])
            arrow = rendering.make_triangle(5, angle=goal_ang)
            arrow.set_color(1.0, 0., 0.)
            arrow_xform = rendering.Transform()
            arrow.add_attr(arrow_xform)
            arrow_xform.set_translation(*self.world.agents[0].state.p_pos + goal_dir)
            self.viewers[i].add_geom(arrow)
            # add formation center to render
            ctr = []
            ctr_xform = []
            ctr = rendering.make_circle(1)
            ctr_xform = rendering.Transform()
            ctr.add_attr(ctr_xform)
            ctr_xform.set_translation(*self.agents[0].agents_ctr)
            self.viewers[i].add_geom(ctr)
            # add previous formation center to render
            ctr_prev = []
            ctr_xform_prev = []
            ctr_prev = rendering.make_circle(1)
            ctr_prev.set_color(0.5, 0.5, 0.5)
            ctr_xform_prev = rendering.Transform()
            ctr_prev.add_attr(ctr_xform_prev)
            ctr_xform_prev.set_translation(*self.agents[0].agents_ctr_prev)
            self.viewers[i].add_geom(ctr_prev)

            # add head to agents
            for e, agent in enumerate(self.agents):
                for j in range(agent.start_ray[0], agent.end_ray[0] + 1):
                    # 105 for compensating square's rendering error
                    if 100 * agent.ray[j][0] < 200:
                        ray_pos = 105 * agent.ray[j][0] * np.array(
                            [np.cos(agent.ray[j][1] + agent.state.p_ang), np.sin(agent.ray[j][1] + agent.state.p_ang)])
                        ray = rendering.make_line(agent.state.p_pos, agent.state.p_pos + ray_pos)
                        ray.set_color(1., 0., 0.)
                        ray_xform = rendering.Transform()
                        ray.add_attr(ray_xform)
                        self.viewers[i].add_geom(ray)
                for j in range(agent.start_ray[1], agent.end_ray[1] + 1):
                    # 105 for compensating square's rendering error
                    if 100 * agent.ray[j][0] < 200:
                        ray_pos = 105 * agent.ray[j][0] * np.array(
                            [np.cos(agent.ray[j][1] + agent.state.p_ang), np.sin(agent.ray[j][1] + agent.state.p_ang)])
                        ray = rendering.make_line(agent.state.p_pos, agent.state.p_pos + ray_pos)
                        ray.set_color(1., 0., 0.)
                        ray_xform = rendering.Transform()
                        ray.add_attr(ray_xform)
                        self.viewers[i].add_geom(ray)
                head = rendering.make_circle(agent.size / 8)
                head_xform = rendering.Transform()
                head.set_color(0.0, .0, 1.0)
                head.add_attr(head_xform)
                displacement = 0.6*agent.size*np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])
                head_xform.set_translation(*agent.state.p_pos + displacement)
                self.viewers[i].add_geom(head)
                label = rendering.make_text(text='%d' % e, font_size=12, x=agent.state.p_pos[0], y=agent.state.p_pos[1], color=(0, 0, 0, 255))
                self.viewers[i].add_label(label)
                dis_btw_agents = rendering.make_text(text=lines_length_text[e], font_size=15,
                                            x= self.world.width // 2 - WINDOW_W // 1.5,
                                            y= self.world.width // 2 - WINDOW_H // 2.0 - 20 * (e + 2),
                                            anchor_x='left',
                                            color=(0, 0, 0, 255))
                self.viewers[i].add_label(dis_btw_agents)

                # for name_num, name in enumerate(reward_names):
                #     agent_reward_text = rendering.make_text(text=name + ' '+str(np.around(reward_dict[name][e], decimals=2)), font_size=15,
                #                                         x= self.world.width // 2 - WINDOW_W // 1.5 + 200 * name_num,
                #                                         y= self.world.width // 2 + WINDOW_H // 2.0 - 20 * (e + 2),
                #                                         anchor_x='left',
                #                                         color=(0, 0, 0, 255))
                #     self.viewers[i].add_label(agent_reward_text)
            time = rendering.make_text(text='time = %f sec' % self.world.time, font_size=15,
                                           x=self.world.width // 2 - WINDOW_W // 1.5,
                                           y=self.world.width // 2 - WINDOW_H // 2.0,
                                           anchor_x='left',
                                           color=(0, 0, 0, 255))
            distance = rendering.make_text(text='distance = %f meters' % self.world.distance, font_size=15,
                                           x=self.world.width // 2 - WINDOW_W // 1.5,
                                           y=self.world.width // 2 - WINDOW_H // 2.0-20,
                                           anchor_x='left',
                                           color=(0, 0, 0, 255))
            self.viewers[i].add_label(distance)
            self.viewers[i].add_label(time)
            # render to display or array
            for line in lines:
                self.viewers[i].add_geom(line)
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))


            '''for j in range(len(self.viewers)):
                self.viewers[i].geoms.pop(-1)'''
        
        return results
    
    def render_origin(self, mode='human', close=False):
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)

            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            self.comm_geoms = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                entity_comm_geoms = []

                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                else:
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from . import rendering

            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(
                pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)

            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx

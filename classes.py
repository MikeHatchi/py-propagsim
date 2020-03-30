from numpy.random import choice, uniform
from numpy import array, concatenate, where
from itertools import compress
from utils import get_least_severe_state, get_move_proba_matrix
from multiprocessing import Pool, cpu_count


class State:
    def __init__(self, id, name, contagiousity, sensitivity, severity):
        """ A state can be carried by an agent. It makes the agent accordingly contagious, 
        sensitive and in a severe state.
        
        :param id: id of the state
        :type id: Integer (int)
        :param name: name of the state
        :type name: String (str)
        :param contagiousity: parameter positively correlated to the probability of 
            infecting other agents in the same environment if one agent is carrying this state
        :type contagiousity: float between 0 and 1
        :param sensitivity: parameter positively correlated to the probability of getting infected by 
            another agent if one agent carrying this state is in the same environment than another agent carrying 
            a state having contagiousity > 0
        :type sensitivity: float between 0 and 1
        :param severity: measure the severity of the state
        :type severity: float between 0 and 1
        """
        self.id = id
        self.name = name
        self.contagiousity = contagiousity
        self.sensitivity = sensitivity
        self.severity = severity

    def __str__(self):
        return self.name

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_contagiousity(self):
        return self.contagiousity

    def get_sensitivity(self):
        return self.sensitivity

    def get_severity(self):
        return self.severity


class Agent:
    def __init__(self, id, p_move, transitions, states, durations, current_state, home_env_id):
        """ An agent can move from environment to environment and get infected by another 
        agent being simoultaneously in the same environment.
        
        :param id: id of the agent
        :type id: Integer (int)
        :param p_move: Probability for an agent to move if a move is made in the <Map> where it is
        :type p_move: float between 0 and 1
        :param transitions: Markovian transition matrix between the `states`
        :type transitions: numpy array of shape (#states, #states) with rows summing up to 1
        :param states: states in which an agent can be
        :type states: list of <State>
        :param durations: durations of the states if the agent move to them
        :type durations: iterable of length #states containing strictly positive integers
        :param current_state: state of the agent at its initialization
        :type current_state: <State>
        :param home_env_id: if of the home environment of the agent
        :type home_env_id: int
        """
        self.id = id
        self.p_move = p_move
        self.transitions = transitions
        self.states = states
        self.durations = durations
        self.current_state = current_state
        self.home_env_id = home_env_id
        self.current_env_id = home_env_id
        # Define dicts for `states`
        self.name2state = {state.get_name(): state for state in states}
        self.name2index = {state.get_name(): i for i, state in enumerate(states)}
        self.index2state = {i: state for i, state in enumerate(states)}
        self.state_names = {state.get_name() for state in states}
        # Define variables for `current_state`
        self.n_states = len(states)
        self.current_state_age = 0
        self.current_state_index = self.name2index.get(current_state.get_name())
        self.least_severe_state = get_least_severe_state(states)


    def get_id(self):
        return self.id


    def get_p_move(self):
        return self.p_move


    def get_home_env_id(self):
        return self.home_env_id


    def get_current_env_id(self):
        return self.current_env_id


    def set_current_env_id(self, env_id):
        self.current_env_id = env_id


    def get_state(self):
        return self.current_state


    def get_least_severe_state(self):
        return self.least_severe_state


    def set_state(self, state):
        """ Force agent to move to given state """
        self.current_state = state
        self.current_state_age = 0
        self.current_state_index = self.name2index.get(state.get_name())


    def get_infected(self):
        """ if the agent gets infected, it jumps to the state having the lowest >0 severity if 
        its current state has severity 0, otherwise it stays to the same state, for 
        preventing absurd behavior like agent getting to a less severe state after infection """
        if self.current_state.get_severity() < self.least_severe_state.get_severity():
            self.set_state(self.least_severe_state)


    def get_next_state(self, state):
        """ randomly sample the state following `state` according to `self.transitions` """
        probas = self.transitions[self.name2index.get(state.get_name()),:]
        index_new_state = choice(self.n_states, 1, p=probas.flatten())
        new_state = self.index2state.get(index_new_state[0])
        return new_state


    def forward(self):
        """ Go one time-step forward and update agent state accordingly: if the current state of 
        the agent reaches its end, agent moves to its next state"""
        self.current_state_age += 1
        # If current state reached age to change, move to next state
        if self.current_state_age >= self.durations[self.current_state_index]:
            new_state = self.get_next_state(self.current_state)
            self.set_state(new_state)



class Environment:
    def __init__(self, id, position, attractivity, unsafety, agents):
        """An environment is figuratively a place where several agents can be together and possibly get 
        infected from an infected agent in the environment.
        An environment has also a geographic `position` (Euclidean coordinates) and an `attractivity` influencing the 
        probability of the agents in other environments to move in this environment.
        
        :param id: id of the environment
        :type id: Integer (int)
        :param position: position of the environment
        :type position: iterable of length 2 containing non-imaginary numerical values
        :param attractivity: attractivity of the environment
        :type attractivity: positive numerical value
        :param unsafety: unsafety of the environment (positively correlated to the probability than contagions happen in the environment)
        :type unsafety: numerical value between 0 and 1
        :param agents: agents initially belonging to the environment. This environment will be their home environment
        :type agents: list of <Agent>
        """
        self.id = id
        self.position = position
        self.attractivity = attractivity
        self.unsafety = unsafety
        self.agents = agents

    def get_id(self):
        return self.id

    def get_position(self):
        return self.position

    def get_attractivity(self):
        return self.attractivity

    def get_unsafety(self):
        return self.unsafety

    def set_unsafety(self, unsafety):
        self.unsafety = unsafety

    def get_agents_id(self):
        return [agent.get_id() for agent in self.agents]

    def update_agent_states(self):
        """ update the state of the agent in the environment by activing contagion. The probability for an agent to get infected is:
        (greatest contagiousity among agents in environment) * (sensitivity of agent)
        """
        greatest_contagiousity = max([agent.get_state().get_contagiousity() for agent in self.agents])
        if greatest_contagiousity == 0: 
            return  # no update if no agent in the environment is contagious
        for agent in self.agents:  # unnecessary to parallelize, there shouldn't be too many agents in a single environment
            proba_infection = greatest_contagiousity * agent.get_state().get_sensitivity() * self.unsafety
            draw = uniform()
            if draw < proba_infection:
                agent.get_infected()

    def add_agent(self, agent, update=True):
        """ add `agent` to the environment. `update`: if to proceed to contagion within the environment or not. """
        self.agents.append(agent)
        if update:
            self.update_agent_states()

    def remove_agent(self, agent_id):
        """ Remove agent which id (caution!) is `agent_id`. No update possible here since 
        removing an agent from an environment doesn't change the state of the remaining agents """
        self.agents = [agent for agent in self.agents if agent.get_id() != agent_id]

        
class Map:
    def __init__(self, environments, agents):
        """ A map contains a list of `environments`, `agents` and an implementation of the 
        way agents can move from an environment to another.
        It also contains methods to get the current repartition of agents among the environments.
        
        :param environments: environments contained in the map
        :type environments: list of <Environment>
        :param agents: agents contained in the map
        :type agents: list of <Agent>
        """
        self.environments = environments
        self.agents = agents
        self.n_envs = len(environments)
        self.n_agents = len(agents)
        # Define dicts to access own environments and agents
        self.id2env = {environment.get_id(): environment for environment in environments}
        self.id2agents = {agent.get_id(): agent for agent in agents}
        
        self.pos_agents_arr = array([self.id2env.get(ind.get_home_env_id()).get_position() for ind in agents])
        self.pos_envs_arr = array([environment.get_position() for environment in environments])
        self.attractivity_arr = array([environment.get_attractivity() for environment in environments])

        self.move_proba_matrix = get_move_proba_matrix(self.pos_envs_arr, self.pos_agents_arr, self.attractivity_arr)


    def move_agent_env(self, agent, environment, update=True):
        """ Move `agent` to `environment`. 
        `update`: if to update its new environment state after the move """
        current_env = self.id2env.get(agent.get_current_env_id())
        current_env.remove_agent(agent.get_id())
        environment.add_agent(agent, update=update)
        agent.set_current_env_id(environment.get_id())


    def move_single_agent(self, i):
        current_agent = self.agents[i]
        probas_new_env = self.move_proba_matrix[:,i].flatten()
        ind_new_env = int(choice(self.n_envs, 1, p=probas_new_env))
        new_env = self.environments[ind_new_env]
        self.move_agent_env(current_agent, new_env)


    def move_home(self, agent, update=False):
        """ Move `agent` to its home environment. 
        `update`: if to update its home environment state after the move """
        if agent.get_current_env_id() == agent.get_home_env_id():
            return
        environment = self.id2env.get(agent.get_home_env_id())
        self.move_agent_env(agent, environment, update)


    def move_home_ind(self, i):
        self.move_home(self.agents[i])


    @classmethod
    def get_single_proba_move(cls, agent):
        return agent.get_p_move() * (1 - agent.get_state().get_severity())


    def make_move(self):
        """ Select agents to move according to their probability to move and then 
        move these to an environment according to `move_proba_matrix`. If an agent is not selected for a move, 
        it's considered to return (or stay) home during this move """
        # Select agents who make a move
        draw = uniform()
        
        pool = Pool(cpu_count())
        probas_move = pool.map(Map.get_single_proba_move, self.agents)
        probas_move = array(probas_move)
        inds_agents2move = where(probas_move >= draw)[0]

        for i in inds_agents2move:
            pool.apply_async(self.move_single_agent, args=(self, i))
        
        # Move back home agents who didn't move
        inds_agents2home = list(set(list(range(self.n_agents))) - set(inds_agents2move))

        for i in inds_agents2home:
            pool.apply_async(Map.move_home_ind, args=(self, i))

        pool.close()


    def all_home(self):
        """ Move all agents to their home environment """
        pool = Pool(cpu_count())
        for i in range(self.n_agents):
            pool.apply_async(Map.move_home_ind, args=(self, i))
        pool.close()


    def get_repartition(self):
        """ returns repartition of agents by environment as dict: env_id => [agent_id in environment] """
        return {environment.get_id(): environment.get_agents_id() for environment in self.environments}
            
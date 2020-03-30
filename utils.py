from numpy import isinf
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import warnings


warnings.filterwarnings('ignore', category=RuntimeWarning) 


def get_least_severe_state(states):
    """ Get the state that has the least severity > 0 """
    least_severe_state = None
    for state in states:
        if state.severity == 0:
            continue
        if least_severe_state is None:
            least_severe_state = state
        elif state.get_severity() < least_severe_state.get_severity():
            least_severe_state = state
    if least_severe_state is None:  # all states have severity 0
        least_severe_state = states[0]
    return least_severe_state


def get_move_proba_matrix(pos_envs_arr, pos_agents_arr, attractivity_arr):
    """ Compute for each agent the probability repartition of the environments for a next move (it it happens)
    If an agent moves, then to a different environment than its current one: the probability for an agent to move 
    to its current environment is 0.
    Otherwise, the probability for an agent to move to a environment ~ attractivity environment) * 1/ dist environment, base_position(agent))
    
    :param pos_envs_arr: positions of the environments
    :type pos_envs_arr: numpy array of shape (#environments, 2)
    :param pos_agents_arr: positions of the agents
    :type pos_agents_arr: numpy array of shape (#agents, 2)
    :param attractivity_arr: attractivity of the environments
    :type attractivity_arr: numpy array of shape (#environments)
    :return: array for which each column represent the probability 
        repartition among the environments to move there in a possible next move
    :rtype: numpy array of dimension (#environments, #agents)
    """
    mat = 1 / cdist(pos_envs_arr, pos_agents_arr, 'euclidean')
    mat[isinf(mat)] = 0  # zero probability to move to the same place (where 1/dist is inf)
    mat *= attractivity_arr[:, None]  # col-wise multiplication
    mat /= norm(mat, ord=1, axis=0, keepdims=True)  # nor each col s.t. sums up to 1 (proba repartition)
    return mat
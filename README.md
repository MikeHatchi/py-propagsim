This is the implementation in Python of a propagation model that we propose to call *EAST* as *Environment Agent State Transitions*.

# Model description
## Basic objects
Basically, this model considers three types of object:
1. **Environment**. An *environment* can contain 0, 1 or many *agent*s at any time. It has also a *position* (Euclidean coordinates).
2. **Agent**. An agent has one (and only one) *state* at a given moment, and is also in one (and only one) environment. The *environment* where an *agent* is initially is considered to be its *home_environment*.
3. **State**. A state (which could be healthy, infected, ...) has predefined parameters *severity*, *contagiousity* and *sensitivity*, all in (0, 1).

## Contagion
A contagion happens within a environment, when it contains several agents at the same time and one of them is contagious.
When an agent has a state with *contagiousity* > 0, then the other agents in the same environment can get infected. 
The probability of *Agent_1* to contaminate *Agent_2* in *environment* is given by:

<img src="https://render.githubusercontent.com/render/math?math=p = contagiousity(state(Agent_1)) \times sensitivity(state(Agent_2)) \times unsafety(environment)">

**NB**: 
* The highest contagiousity in the environment is taken to compute *p*.
* The *unsafety* of a *environment* measures how a environment is unsafe for contagion (social distancing respected or not inside etc.).

If *Agent_2* gets infected, it gets to its own state having the least strictly positive *severity* (it can't jump directly to a more severe state).

## State transition
A *state transition matrix* and *state durations* are attached to each *agent*. The *state transition matrix* is a Markovian matrix describing the transition probabilities between the states an agent can take. The *state durations* are the duration of each state. If a agent is in a given *state*, it will switch to another one sampled according to its *state transition matrix*.

**NB**: Different *agent*s can share the same states and the same *state transition matrix* but have different *state durations*, or have have the same *state*, the same *state durations* but a different *state transition matrix*, or etc.

## Moves
A move consists in moving *agent*s to other environments. When a move is done, all *agent*s are concerned. It happens it two steps:
1. Selecting *agent*s that will move.
2. Moving the selected *agent*s to their new environments.

### Agent selection
The probability of an *agent* to be selected for a move is:

<img src="https://render.githubusercontent.com/render/math?math=p = proba\_move(agent) \times (1 - severity(state(agent)))">
 

The first factor represents the mobility of the *agent* so to say. The second factor represents the fact that the more severe the state of an *agent*, the less the probability that it will move.

### Environment selection
The *environment* where to move a selected *agent* is sampled according to a probability

<img src="https://render.githubusercontent.com/render/math?math=p \~ distance(home\_env(agent), environment) \times attractivity(environment)">

**NB**: 
* a limitation of this model is that the attractivity of each *environment* is the same for all *agent*. An extension / refinement of this model would be to have *environment* attractivities personalized by agent.
* The distance is always computed from the *home_environment* of an agent, not from its *current_cell*. An *agent* is considered wandering around its *home_environment*
* The *agent*s not selected for a move will be moved to their *home_environment* afterward

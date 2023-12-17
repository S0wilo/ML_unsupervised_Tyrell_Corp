from agent import Agent
import numpy as np


def tyrell_corp_policy(agent: Agent) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    actions = ["left", "right", "none"]

    #change position if agent is placed in the first or last position and there is nothing here
    if agent.position == 0 and agent.known_rewards[0] == 0:
        return "right"
    if agent.position == 7 and agent.known_rewards[7] == 0:
        return "left"

    #change position if the max value of the known rewards is lower than the average reward value (50)
    max_val = max(agent.known_rewards)
    if max_val < 50:
        #change position in the direction of the unknown part of the array
        mean_left = np.mean(agent.known_rewards[:2])
        mean_right = np.mean(agent.known_rewards[-2:])
        if mean_left < mean_right:
            return "left"
        return "right"

    #change position if the agent is not placed on the best rewards of the array
    idx = np.where(agent.known_rewards == max(agent.known_rewards))[0]
    if idx < agent.position:
        return "left"
    if idx > agent.position:
        return "right"

    return "none"

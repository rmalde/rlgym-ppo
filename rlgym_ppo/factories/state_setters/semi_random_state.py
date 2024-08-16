from rlgym_sim.utils.state_setters import StateSetter, RandomState, DefaultState, StateWrapper

import random

class SemiRandomState(StateSetter):
    def __init__(self, p_random):
        super().__init__()
        self.p_random = p_random
        self.random_state = RandomState()
        self.default_state = DefaultState()
        
    def reset(self, state_wrapper: StateWrapper):
        # generate random number
        random_number = random.random()
        if random_number < self.p_random:
            self.random_state.reset(state_wrapper)
        else:
            self.default_state.reset(state_wrapper)





#!/usr/bin/env python3
"""Perform forward algorithm for hidden markov model"""


import numpy as np


def forward(Oberservation, Emission, Transition, Initial):
    """
    Perform forward algorithm for hidden markov model
    Observation: index of observation. (num observations,) ndarray
    Emission: emission probability. (hidden states, all possible observations)
    Transition: transition probabilities (hidden states, hidden states)
    Initial: initial state probailities. (hidden states, 1)
    """
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Initial, np.ndarray)):
        return None
    if ((len(Observation.shape) != 1 or Observation.shape[0] < 1 or
         (len(Emission.shape) != len(Transition.shape) != len(Initial.shape)
          != 2) or (Transition.shape[0] != Transition.shape[1] !=
                    Emissions.shape[0] != Initial.shape[0]) or
         Transition.shape[0] < 1)):
        return None
    if (not np.where(np.isclose(Transition.sum(axis=1), 1), 1, 0).any() or
        not np.where(np.isclose(Initial.sum(axis=1), 1), 1, 0).any() or
        not np.where(np.isclose(Emission.sum(axis=1), 1), 1, 0).any())
        return None
    

from comma.individual import Individual
import numpy as np
from pathlib import Path

def test_choose_actions_on_lockdown():
    """
    Unit test for the `choose_actions_on_lockdown` method of the `Individual` class.

   The test checks whether the method returns correct types for the `actions` and
   `action_probs` outputs, and whether these outputs follow the expected properties:
    - `actions` is an array of booleans.
    - `action_probs` is a numpy array of probabilities (values between 0 and 1).
    - Both `actions` and `action_probs` should have the same length.
    - `actions` array length should be 10 (assuming there are 10 possible actions).

    """

    dir_params = Path("parameters/")  # specify the path to your parameters
    lockdown = 'easy'  # specify the type of lockdown

    # Create an individual with id 0
    individual = Individual.populate(1, dir_params)

    actions, action_probs = individual[0].choose_actions_on_lockdown(lockdown)

    assert isinstance(actions, np.ndarray), 'actions should be a numpy array'
    assert isinstance(action_probs, np.ndarray), 'action_probs should be a numpy array'

    assert len(actions) == 10, 'actions array should have length 10'
    assert np.all((actions == False) | (actions == True)), 'all actions should be False or True'

    assert np.all((0 <= action_probs) & (action_probs <= 1)), 'all action_probs should be between 0 and 1'
    assert len(action_probs) == len(actions), 'action_probs and actions should have the same length'

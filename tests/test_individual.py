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

def test_populate():
    """
    Unit test for the `populate` method of the `Individual` class.

    The test checks whether `Individual._features` has the correct columns.
    Additionally, it ensures that the values in these columns are either 0 or 1.

    """
    dir_params = Path("parameters/")  # specify the path to your parameters

    # Populate with one individual for the test
    individuals = Individual.populate(1, dir_params)

    expected_cols = [
        'baseline',
        'age_group__1',
        'age_group__2',
        'age_group__3',
        'age_group__4',
        'gender_f',
        'gender_m',
        'education_high',
        'education_low',
        'education_medium',
        'unemployed_no',
        'unemployed_yes',
        'have_partner_no',
        'have_partner_yes',
        'depressed_no',
        'depressed_yes',
        'children_presence_no',
        'children_presence_yes',
        'housing_financial_difficulties_no',
        'housing_financial_difficulties_yes',
        'selfrated_health_average',
        'selfrated_health_good',
        'selfrated_health_poor',
        'critical_job_no',
        'critical_job_yes'
    ]

    assert list(Individual._features.columns) == expected_cols, 'Columns in Individual._features returned by populate() are incorrect'

    for individual in individuals:
        assert np.all((individual.get_features() == 0) | (individual.get_features() == 1)), \
               'Values in Individual._features should be either 0 or 1'


def test_populate_ipf():
    """
    Unit test for the `populate_ipf` method of the `Individual` class.

    The test checks whether the DataFrame returned by `populate_ipf` has the correct columns.
    Additionally, it ensures that the values in these columns are either 0 or 1.

    """
    dir_params = Path("parameters/")  # specify the path to your parameters

    # Call populate_ipf for testing
    df = Individual.populate_ipf(1, dir_params)

    expected_cols = [
        'baseline',
        'age_group__1',
        'age_group__2',
        'age_group__3',
        'age_group__4',
        'gender_f',
        'gender_m',
        'education_high',
        'education_low',
        'education_medium',
        'unemployed_no',
        'unemployed_yes',
        'have_partner_no',
        'have_partner_yes',
        'depressed_no',
        'depressed_yes',
        'children_presence_no',
        'children_presence_yes',
        'housing_financial_difficulties_no',
        'housing_financial_difficulties_yes',
        'selfrated_health_average',
        'selfrated_health_good',
        'selfrated_health_poor',
        'critical_job_no',
        'critical_job_yes'
    ]

    assert list(df[0].get_features().index) == expected_cols, 'Columns in the dataframe returned by populate_ipf() are incorrect'

    assert np.all((df[0].get_features() == 0) | (df[0].get_features() == 1)).all(), 'Values in the dataframe returned by populate_ipf should be either 0 or 1'
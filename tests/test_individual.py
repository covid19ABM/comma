import pytest
from comma.hypothesis import Hypothesis
from comma.individual import Individual
import numpy as np
from pathlib import Path


class TestIndividual:
    @pytest.fixture
    def dir_params(self):
        return Path("parameters/")

    @pytest.fixture
    def expected_actions(self):
        return np.array(
            [
                "maintain_physical_distance",
                "stay_at_home",
                "exercise",
                "socialise",
                "seek_help",
                "negative_coping",
            ]
        )

    @pytest.fixture
    def expected_status(self):
        return np.array(-12.510000000000002)

    @pytest.fixture
    def lockdown(self):
        return "easy"

    @pytest.fixture
    def expected_cols(self):
        return [
            "baseline",
            "age_group__1",
            "age_group__2",
            "age_group__3",
            "age_group__4",
            "gender_f",
            "gender_m",
            "education_high",
            "education_low",
            "education_medium",
            "unemployed_no",
            "unemployed_yes",
            "have_partner_no",
            "have_partner_yes",
            "depressed_no",
            "depressed_yes",
            "children_presence_no",
            "children_presence_yes",
            "housing_financial_difficulties_no",
            "housing_financial_difficulties_yes",
            "selfrated_health_average",
            "selfrated_health_good",
            "selfrated_health_poor",
            "critical_job_no",
            "critical_job_yes",
        ]

    @pytest.fixture
    def seed(self):
        return np.random.SeedSequence(0)

    def test_take_actions(
        self, dir_params, expected_actions, expected_status, lockdown, seed
    ):
        """
        Test for the `take_actions` method of the individual class.

        Validates:
        - Matrix multiplication result.
        - Correct actions are chosen based on input.
        - Status after action is as expected.
        """

        # specify the actions
        actions = np.array(
            [False, True, True, True, True, False, True, True, False, False]
        )

        # specify the actions effects
        actions_effects = Hypothesis.read_hypotheses(
            dir_params, set([lockdown]), "actions"
        )
        individual = Individual.populate(1, dir_params, rng=np.random.default_rng(seed))
        individual[0].take_actions(actions, actions_effects[lockdown])

        individual[0].chosen_actions = actions
        actual_actions = individual[0].get_actions()
        actual_status = individual[0]._status

        assert np.all(
            actual_actions == expected_actions
        ), "actions chosen should match the expected"
        assert np.all(
            actual_status == expected_status
        ), "status after action taken should be equal to 0."

    def test_choose_actions_on_lockdown(self, dir_params, lockdown, seed):
        """
        Unit test for the `choose_actions_on_lockdown`
        method of the `Individual` class.
        The test checks whether the method returns correct
        types for the `actions` and `action_probs` outputs,
        and whether these outputs follow the expected properties:
        - `actions` is an array of booleans.
        - `action_probs` is a numpy array of probabilities (between 0 and 1).
        - Both `actions` and `action_probs` should have the same length.
        - `actions` array length should be 10.
        """

        current_lockdown = Hypothesis.read_hypotheses(
            dir_params, set([lockdown]), "lockdown"
        )
        # Create an individual with id 0
        individual = Individual.populate(1, dir_params)

        actions, action_probs = individual[0].choose_actions_on_lockdown(
            current_lockdown[lockdown], rng=np.random.default_rng(seed)
        )

        assert isinstance(actions, np.ndarray), "actions should be a numpy array"
        assert isinstance(
            action_probs, np.ndarray
        ), "action_probs should be a numpy array"

        assert len(actions) == 10, "actions array should have length 10"
        assert np.all(
            (actions == 0) | (actions == 1)
        ), "all actions should be False or True"

        assert np.all(
            (0 <= action_probs) & (action_probs <= 1)
        ), "all action_probs should be between 0 and 1"
        assert len(action_probs) == len(
            actions
        ), "action_probs and actions should have the same length"

    def test_populate(self, dir_params, expected_cols, seed):
        """
        Unit test for the `populate` method of the `Individual` class.

        The test checks whether `Individual._features` has the correct columns.
        It also ensures that the values in these columns are either 0 or 1.
        """

        # Populate with one individual for the test
        individuals = Individual.populate(
            1, dir_params, rng=np.random.default_rng(seed)
        )

        assert (
            list(individuals[0]._features.index) == expected_cols
        ), "Columns in Individual._features are incorrect"

        for individual in individuals:
            assert np.all(
                (individual.get_features() == 0) | (individual.get_features() == 1)
            ), "Values in Individual._features should be either 0 or 1"

    def test_populate_ipf(self, dir_params, expected_cols, seed):
        """
        Unit test for the `populate_ipf` method of the `Individual` class.

        The test checks whether the DataFrame returned by
        `populate_ipf` has the correct columns.
        It also ensures that the values in these columns are either 0 or 1.
        """

        # Call populate_ipf for testing

        df = Individual.populate_ipf(1, dir_params, rng=np.random.default_rng(seed))

        assert (
            list(df[0].get_features().index) == expected_cols
        ), "Columns in the dataframe returned by populate_ipf() are incorrect"

        assert np.all(
            (df[0].get_features() == 0) | (df[0].get_features() == 1)
        ).all(), (
            "Values in the dataframe returned "
            "by populate_ipf should be either 0 or 1"
        )

    def test_actions_when_positive(self, dir_params, lockdown, seed):
        """
        Test that modify_policy_when_infected returns 'stay at home' action
        """
        policy = Hypothesis.read_hypotheses(dir_params, set([lockdown]), "lockdown")
        agent = Individual.populate_ipf(1, dir_params, rng=np.random.default_rng(seed))
        lockdown_new = agent[0].modify_policy_when_infected(policy[lockdown])
        _, action_probs = agent[0].choose_actions_on_lockdown(lockdown_new)
        action_out = agent[0].get_actions()
        assert action_out[0] == "stay_at_home", (
            f"Expected 'stay at home' " f"got {action_out[0]}"
        )
        assert action_probs.round()[2] == 1, (
            f"Expected Probability ~ 1 " f"got {action_probs.round()[2]}"
        )

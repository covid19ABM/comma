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
                "maintain_social_distance",
                "exercise",
                "feel_socially_connected",
                "seek_help_from_friends",
                "positive_coping",
                "feel_isolated",
            ]
        )

    @pytest.fixture
    def expected_status(self):
        return np.array(7.5169999999999995)

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
            "education_unknown",
            "unemployed_no",
            "unemployed_yes",
            "have_partner_no",
            "have_partner_yes",
            "have_partner_unknown",
            "depressed_no",
            "depressed_yes",
            "depressed_unknown",
            "children_presence_no",
            "children_presence_yes",
            "children_presence_unknown",
            "housing_financial_difficulties_no",
            "housing_financial_difficulties_yes",
            "housing_financial_difficulties_unknown",
            "selfrated_health_average",
            "selfrated_health_good",
            "selfrated_health_poor",
            "selfrated_health_unknown",
            "critical_job_no",
            "critical_job_yes",
            "critical_job_unknown",
            "bmi_underweight",
            "bmi_normalweight",
            "bmi_overweight",
            "bmi_obese",
            "bmi_unknown",
            "livesalone_no",
            "livesalone_yes",
            "livesalone_unknown",
            "income_median_above",
            "income_median_below",
            "income_median_unknown",
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

        # specify the actions effects
        actions_effects = Hypothesis.read_hypotheses(
            dir_params, set([lockdown]), "actions"
        )
        individual = Individual.populate(1, dir_params, rng=np.random.default_rng(seed))
        # specify the actions
        individual[0].chosen_actions = np.array(
            [False, True, True, True, True, False, True, True, False]
        )
        individual[0].take_actions(actions_effects[lockdown])

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

        individual[0].choose_actions_on_lockdown(
            current_lockdown[lockdown], rng=np.random.default_rng(seed)
        )

        assert isinstance(
            individual[0].chosen_actions, np.ndarray
        ), "actions should be a numpy array"

        assert (
            len(individual[0].chosen_actions) == 9
        ), "actions array should have length 9"

        assert np.all(
            (individual[0].chosen_actions == 0) | (individual[0].chosen_actions == 1)
        ), "all actions should be False or True"

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
        Test that modify_policy_when_infected returns 'be sedentary' action
        """
        policy = Hypothesis.read_hypotheses(dir_params, set([lockdown]), "lockdown")
        agent = Individual.populate_ipf(1, dir_params, rng=np.random.default_rng(seed))
        lockdown_new = agent[0].modify_policy_when_infected(policy[lockdown])
        agent[0].choose_actions_on_lockdown(lockdown_new)
        action_out = agent[0].get_actions()
        assert (
            action_out[0] == "be_sedentary"
        ), f"Expected 'be_sedentary' got {action_out[0]}"

    def test_long_covid(self, dir_params, seed):
        """
        Test over a large number of instances that proportions
        are always ~ 20%
        """
        num_agents = 10000
        long_covid_cases = 0
        agents = Individual.populate_ipf(
            num_agents, dir_params, rng=np.random.default_rng(seed)
        )

        for agent in agents:
            if agent.is_long_covid():
                long_covid_cases += 1

        proportion = long_covid_cases / num_agents
        expected = 0.20
        error_margin = 0.01
        assert proportion == pytest.approx(expected, abs=error_margin)

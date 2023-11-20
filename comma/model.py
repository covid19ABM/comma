"""Model class definition
"""
from comma.individual import Individual
from comma.hypothesis import Hypothesis
import pandas as pd
import numpy as np
from tqdm import tqdm


class Model:
    def __init__(
        self, size: int, dir_params: str, use_ipf: bool = False, seed=None
    ) -> None:
        self.simulation_id: int = None
        self.current_step: int = 0  # keep track of the current simulation step
        self.lockdown_status: dict = {}
        self.dir_params: str = dir_params
        # self.date_format = "%Y-%m-%d"
        self.cumulative_status = dict()
        if seed is not None:
            seed_value = np.random.SeedSequence(seed)
            self.rng = np.random.default_rng(seed_value)
        else:
            self.rng = np.random.default_rng(None)

        Hypothesis.validate_param_file(self.dir_params)
        if use_ipf:
            self.agents = Individual.populate_ipf(size, self.dir_params, self.rng)
        else:
            self.agents = Individual.populate(size, self.dir_params, self.rng)

    def update_covid_counter(self):
        """
        Update the days_since_positive counter
        for agents with covid_status of 1.
        Returns:
            None. This function updates the agent's covid counter in place.
        """
        for agent in self.agents:
            if agent.covid_status == 1:
                agent.days_since_positive += 1

    def get_recovered_individuals(self) -> list[int]:
        """
        Get the indices of agents who are recovered from COVID-19

        The function finds positive agents and checks if they are
        recovered based on the number of days since they tested +ve.

        Returns:
            recovered (list): List of indices of recovered agents

        """
        positives = [
            index for index, agent in enumerate(self.agents) if agent.covid_status == 1
        ]
        recovered = []
        for i in positives:
            if self.agents[i].is_recovered():
                recovered.append(i)

        return recovered

    def step(
        self, lockdown: pd.DataFrame, action_effects: pd.DataFrame, new_infected: int
    ) -> None:
        """Actions to be performed in each step.

        Args:
            lockdown (pd.DataFrame): lockdown dataframe
            action_effects (pd.Dataframe): actions dataframe
            new_infected (int): number of new infected
        Returns:
            actions: array of booleans
            action_probs: array of probabilities
        """
        # update counter
        self.update_covid_counter()
        # check recovery
        recovered_idx = self.get_recovered_individuals()

        if recovered_idx:
            for i in recovered_idx:
                # if recovered reset their covid status
                self.agents[i].covid_status = 0
                # and reset the counter
                self.agents[i].days_since_positive = 0

        # extract agents who are negative
        negative_agents = [agent for agent in self.agents if agent.covid_status == 0]

        # make some of them positive (selected randomly)
        random_rng = np.random.default_rng(None)
        newly_infected_agents = random_rng.choice(
            negative_agents, new_infected, replace=False
        )

        # mark selected agents as infected (covid_status = 1)
        # and update counter of positive days for positive people
        for agent in newly_infected_agents:
            agent.covid_status = 1
            agent.days_since_positive = 1

        for agent in self.agents:
            if agent.covid_status == 0:
                # choose actions based on lockdown
                agent.choose_actions_on_lockdown(lockdown, rng=self.rng)
                # take those actions, and compute their effect on mental health
            else:
                # positive agents stay at home
                lockdown_updated = agent.modify_policy_when_infected(lockdown)
                agent.choose_actions_on_lockdown(lockdown_updated, rng=self.rng)
                # depending on lockdown staying at home
                # has certain consequences on mental health
            agent.take_actions(action_effects)

    def update(self, lockdown: str, step: int) -> None:
        """
        Update mental health status at every step given actions

        Args:
            lockdown (str): lockdown type
            step (int): step of the simulation
        """
        if self.current_step == 0:
            delta_mh = 0  # it's day 0, so no incremental change
            self.cumulative_status[step] = [
                (
                    lockdown,
                    agent.id,
                    delta_mh,
                    agent.get_status(),
                    agent.covid_status,
                    agent.days_since_positive,
                )
                for agent in self.agents
            ]
        else:
            # from step 1++, sum the agent's status with the previous status
            agent_statuses = []
            for agent in self.agents:
                last_status = [
                    status
                    for status in self.cumulative_status[step - 1]
                    if status[1] == agent.id
                ][0][-3]
                delta_mh = agent.get_status()  # this is the incremental effect
                mu, sigma = 0.002, 0.0005
                # this is the baseline effect when no action is taken
                # or when action effects are canceled out
                update_rng = np.random.default_rng(None)
                baseline = update_rng.normal(mu, sigma)
                new_status = (
                    lockdown,
                    agent.id,
                    delta_mh,
                    (last_status + delta_mh) - baseline,
                    agent.covid_status,
                    agent.days_since_positive,
                )
                agent_statuses.append(new_status)
            self.cumulative_status[step] = agent_statuses

    def report(self, out_path: str) -> None:
        """
        Collect data recorded at the end of the simulation
        and exports it as csv file.

        Args:
            out_path (str): File path of the output file
        """
        status_data = []

        for step_id, agent_statuses in self.cumulative_status.items():
            for agent_status in agent_statuses:
                (
                    lockdown,
                    agent_id,
                    delta_mh,
                    mh,
                    covid_status,
                    days_since_positive,
                ) = agent_status
                status_data.append(
                    [
                        step_id,
                        lockdown,
                        agent_id,
                        delta_mh,
                        mh,
                        covid_status,
                        days_since_positive,
                    ]
                )

        # create a pandas dataframe to store data from `status_data`
        status_df = pd.DataFrame(
            status_data,
            columns=[
                "step_id",
                "lockdown",
                "agent_id",
                "delta_mental_health",
                "cumulative_mental_health",
                "covid_status",
                "days_since_first_infection",
            ],
        )

        # Export to a csv
        status_df.to_csv(out_path, index=False, sep=";", decimal=",", mode="w+")

    def run(
        self,
        steps: int,
        lockdown_policy: list,
        out_path: str,
        starting_date="2021-02-05",
        location="Groningen",
        real_pop_size=200336,
    ) -> None:
        """Run a simulation

        Args:
            steps(int): Number of steps to run the simulation

            lockdown_policy(str): Type of lockdown policy

            out_path(str): File path of the output file

            location(str): Security region name. This is the name of the city.

            starting_date(str): start date ('YYYY-MM-DD')

            real_pop_size(int): Real size of the population of location
        """
        if steps <= 1:
            raise ValueError("Steps must be more than 1")

        if len(lockdown_policy) != steps:
            raise ValueError(
                "The length of the lockdown list "
                "must be equal to the number of steps"
            )
        # compute time_period
        hypothesis = Hypothesis(starting_date, steps)

        # get new positive cases
        positives = hypothesis.get_positive_cases(location)
        # scale them to the size of the simulated population
        new_cases = hypothesis.scale_cases_to_population(
            positives, real_pop_size, len(self.agents)
        )
        # print(f"scaled cases: {new_cases} \n cases: {positives}")
        # read hypotheses
        lockdown_matrices = hypothesis.read_hypotheses(
            self.dir_params, set(lockdown_policy), "lockdown"
        )

        actions_effects_matrices = hypothesis.read_hypotheses(
            self.dir_params, set(lockdown_policy), "actions"
        )

        # start the simulation
        for step, current_lockdown in tqdm(
            enumerate(lockdown_policy), total=steps, desc="Running simulation"
        ):
            # print(f'new cases: {new_cases[step]}, day: {step}')
            self.simulation_id = step
            self.lockdown_status[step] = current_lockdown
            new_infected = new_cases[step]
            self.step(
                lockdown_matrices[current_lockdown],
                actions_effects_matrices[current_lockdown],
                new_infected,
            )
            self.update(current_lockdown, step)
            self.current_step += 1  # Increment the simulation step
        self.report(out_path)

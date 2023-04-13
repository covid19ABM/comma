"""
Classes for the Mental Health Modelling agent-based model.
These classes are: Features, Agent, State and Population.
"""

from dataclasses import dataclass, fields, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mhm.distributions as dist
import mhm.utils as ut

@dataclass
class Features():
    ''' Features of the agents '''
    __slots__ = ['name', 'gender', 'age', 'age_group', 'education', 'employed',
                 'partnership_status', 'pre_existing_depression',
                 'pre_existing_burnout', 'pre_existing_addiction',
                 'pre_existing_chronic_fatigue', 'parenthood',
                 'living_with_child', 'single_parent', 'housing_difficulties',
                 'finance_difficulties', 'pre_existing_health_issues',
                 'partner_difficulties', 'job_type']
    name: int
    gender: int
    age: int
    age_group: int
    education: str
    employed: str
    partnership_status: str
    pre_existing_depression: int
    pre_existing_burnout: int
    pre_existing_addiction: int 
    pre_existing_chronic_fatigue: int
    parenthood: int
    living_with_child: int
    single_parent: int
    housing_difficulties: str
    finance_difficulties: str
    pre_existing_health_issues: str
    partner_difficulties: str
    job_type: str
    
    def summary(self):
        ''' Generate a summary of the features '''
        
        for field in fields(self):
            print(f'{field.name}: {getattr(self, field.name)}')

@dataclass
class State():
    ''' Mutable states of the agents '''
    employed: int
    infected: int
    mental_health: int
    n_contacts: int


@dataclass
class Agent():
    ''' Represents an agent in our small world '''
    features: Features
    state: State
    # go to work
    def go_to_work(self):
        ''' The agent is going to work '''
        self.state.n_contacts += 3
        self.state.mental_health += 1      
    def socialise(self):
        ''' The agent is socialising with other people '''
        self.state.n_contacts += 5
        self.state.mental_health += 3 
    def summary(self):
        ''' Generate a summary of the features '''
        self.features.summary()
        print(f'n_contacts: {self.state.n_contacts}')


def make_population(parameters):
    ''' Generate a population of agents 
    
    Parameters
    ----------
    parameters : dict
        Parameters of the agents.

    Returns
    -------
    people : list of agents in the form of dataclasses of the Agent type
        Agents population.
    '''
    
    n_people = parameters['n']
    gd = dist.generate_gender_distribution(n_people=n_people, percentage=parameters['%_gender'])
    ad = dist.generate_age_distribution(n_people=n_people, n_age_groups=parameters['n_age_groups'], min_age=parameters['min_age'], max_age=parameters['max_age'], prob=parameters['age_strata'])
    ea = dist.generate_educational_attainment_distribution(n_people=n_people, low=parameters['low_education'], medium=parameters['medium_education'], high=parameters['high_education'])
    ed = dist.generate_employment_distribution(n_people=n_people, yes=parameters['employed_true'], no_seeking=parameters['employed_false_seeking'], no_other=parameters['employed_false_other'])
    pd = dist.generate_partnership_status_distribution(n_people=n_people, single=parameters['single'], married=parameters['married'], live_in_partner=parameters['live_in_partner'], in_relationship_no_cohabitation=parameters['in_relationship_no_cohabitation'], other=parameters['other_relationship'])
    dd = dist.generate_pre_existing_depression_distribution(n_people=n_people, percentage=parameters['%_depression'])
    pad = dist.generate_pre_existing_addiction_distribution(n_people=n_people, percentage=parameters['%_addiction'])
    pfd = dist.generate_pre_existing_fatigue_distribution(n_people=n_people, percentage=parameters['%_fatigue'])
    cd = dist.generate_child_distribution(n_people=n_people, percentage=parameters['%_child'])
    pbd = dist.generate_pre_existing_burnout_distribution(n_people=n_people, percentage=parameters['%_burnout'])
    lcd = dist.generate_living_with_children_distribution(n_people=n_people, percentage=parameters['%_child_inhouse'])
    hd = dist.generate_health_difficulties_distribution(n_people=n_people, no=parameters['health_difficulties_false'], some=parameters['health_difficulties_some'], many=parameters['health_difficulties_many'])
    fd = dist.generate_finance_difficulties_distribution(n_people=n_people, no=parameters['finance_difficulties_false'], some=parameters['finance_difficulties_some'], many=parameters['finance_difficulties_many'])
    spd = dist.generate_single_parent_distribution(n_people=n_people, percentage=parameters['%_single_parent'])
    hdd = dist.generate_housing_difficulties_distribution(n_people=n_people, no=parameters['housing_difficulties_false'], some=parameters['housing_difficulties_some'], many=parameters['housing_difficulties_many'])
    pdd = dist.generate_partner_difficulties_distribution(n_people=n_people, no=parameters['partner_difficulties_false'], some=parameters['partner_difficulties_some'], many=parameters['partner_difficulties_many'], other=parameters['partner_difficulties_other'])
    jtd = dist.generate_jobType_distribution(n_people=n_people, whiteCollar=parameters['white_collar'], medicalProfessional=parameters['medical_professional'], keyWorker=parameters['key_worker'], other=parameters['other_job'])

    ag = ut.label_age_range(ad)
    
    people = []

    for i in range(n_people):
        features = Features(name=str(i),
                            gender=gd[i], 
                            age=ad[i],
                            age_group= ag[i],
                            education=ea[i], 
                            employed=ed[i], 
                            partnership_status=pd[i], 
                            pre_existing_depression=dd[i], 
                            pre_existing_burnout=pbd[i], 
                            pre_existing_addiction=pad[i], 
                            pre_existing_chronic_fatigue=pfd[i], 
                            pre_existing_health_issues=hd[i], 
                            parenthood=cd[i], 
                            living_with_child=lcd[i], 
                            single_parent=spd[i], 
                            housing_difficulties=hdd[i], 
                            finance_difficulties=fd[i], 
                            partner_difficulties=pdd[i],
                            job_type=jtd[i])
        
        state = State(employed=np.random.choice(2),
                      infected=0,
                      mental_health=2,
                      n_contacts=0)    
        people.append(Agent(features=features, state=state))
    return people


def plot_agents_dist(agent_list, n_cols=5):
    ''' Plot population distributions 
    
    Parameters
    ----------
    agent_list : list
        List of dataclasses of the type Agent

    Returns
    -------
    fig : matplotlib figure
        Barplots of the Population distribution
    '''
    dataframe = ut.make_features_dataframe(agent_list=agent_list)
    n_rows = len(dataframe.columns) // n_cols + 1
    fig = plt.figure(figsize=(20, 20))
    for i, c in enumerate(dataframe.columns):
        subset = dataframe[c].value_counts()
        plt.subplot(n_rows, n_cols, i + 1)    
        plt.bar(subset.index, subset.values)
        plt.title(c)
    return fig


class Population():
    ''' Represents an entire population of agents 
    
    Parameters
    ----------
    parameters : dict
        Agents' parameters
        
    Returns
    -------
    A Population class object containing all agents' characteristics
    '''
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.population = make_population(parameters=self.parameters)
        
    def plot_summary(self):
        plot_agents_dist(agent_list=self.population)
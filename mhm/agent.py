from dataclasses import dataclass, fields, asdict
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


@dataclass
class Features():
    ''' Features of the agents '''
    __slots__ = ['name', 'gender', 'age', 'education', 'employed',
                 'partnership_status', 'pre_existing_depression',
                 'pre_existing_burnout', 'pre_existing_addiction',
                 'pre_existing_chronic_fatigue', 'parenthood',
                 'living_with_child', 'single_parent', 'housing_difficulties',
                 'finance_difficulties', 'pre_existing_health_issues',
                 'partner_difficulties']
    name: int
    gender: int
    age: int
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


def generate_gender_distribution(n_people, percentage):
    ''' Generates gender distribution
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of male over female (e.g., 50 for 50%)

    Returns
    -------
    gender_distribution : numpy array
        Array of 1s and 0s (1 = female, 0 = male) whose proportions depend on `percentage`.
    '''
    gender_distribution = np.ones(n_people)
    gender_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(gender_distribution)       
    return gender_distribution.astype(int)


def generate_age_distribution(n_people, n_age_groups, min_age, max_age, prob):
    ''' Generates age groups distribution in the population
    Parameters
    ----------
    n_people : int
        Size of the population.
    min_age : int
        Minimum age.
    max_age : int
        Maximum age.
    n_age_groups : int
        Number of age groups.
    prob : list   
        Proportion of age prevalence for each age group (e.g., .25). Number of elements must match value in `n_age_groups`.
    
    Returns
    -------
    age_distribution : numpy array
        Array of age distribution in the population. 
        Proportions depend on `prob`.
    '''

    age_values = np.arange(min_age, max_age)  
    age_groups = np.array_split(age_values, n_age_groups)

    age_distribution = []
    for x in range(n_people):
        idx = np.random.choice(range(len(age_groups)), p=prob)
        age_distribution.append(np.random.choice(age_groups[idx]))
    return age_distribution


def generate_educational_attainment_distribution(n_people, low, medium, high):
    ''' Generate distribution of high, medium 
    and low educational attainments in the population.
    
    Parameters
    ----------
    n_people : int
        Size of the population.
    low : float
        Probability of low educational attainment. 
    medium : float
        Probability of medium educational attainment. 
    high : float
        Probability of high educational attainment. 
        
    Returns
    ----------
    educational_attainment : list
        List of type of educational attainments in the population.
    '''
    options = {'Low': low, 
               'Medium': medium,
               'High': high}
    
    choices = list(options.keys())
    weights = list(options.values())
    educational_attainment = random.choices(choices, weights, k=n_people) 
    return educational_attainment


def generate_employment_distribution(n_people, yes, no_seeking, no_other):
    ''' Generate distribution of employment statuses in the population.
    Parameters
    ----------
    n_people : int
        Size of the population.
    yes : float
        Probability of being employed. 
    no_seeking : float
        Probability of not being employed, but looking for a job. 
    no_other : float
        Probability of not being employed, but not looking for a job. 
        
    Returns
    ----------
    employment : list
        List of employment status in the population.
    '''
    options = {'Yes': yes, 
               'No, seeking employment': no_seeking,
               'No, other': no_other}
    
    choices = list(options.keys())
    weights = list(options.values())
    employment = random.choices(choices, weights, k=n_people) 
    return employment


def generate_partnership_statuts_distribution(n_people, single, married, live_in_partner, in_relationship_no_cohabitation, other):
    ''' Generate distribution of partnership statuses in the population.
    Parameters
    ----------
    n_people : int
        Size of the population.
    single : float
        Probability of being single. 
    married : float
        Probability of not married. 
    live_in_partner : float
        Probability of being in a relationship and living together. 
    in_relationship_no_cohabitation : float
        Probability of being in a relationship but not living together.
    other : float
        Probability of being in a partnership status that is none of the above.
        
    Returns
    ----------
    relationship : list
        Distribution of relationship statuses in the population.
    '''
    options = {'Single': single, 
               'Married': married,
               'Live-in partner': live_in_partner,
               'In relationship, no cohabitation': in_relationship_no_cohabitation,
               'Other': other}
    
    choices = list(options.keys())
    weights = list(options.values())
    relationship = random.choices(choices, weights, k=n_people) 
    return relationship


def generate_pre_existing_depression_distribution(n_people, percentage):
    ''' Generates proportion of people 
    that experienced depression in the past year 
    
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    depression_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose 
        proportions depend on `percentage`.
    '''
    depression_distribution = np.ones(n_people)
    depression_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(depression_distribution)       
    return depression_distribution.astype(int)


def generate_pre_existing_burnout_distribution(n_people, percentage):
    ''' Generates proportion of people that 
    experienced burnout in the past year
     
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    burnout_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) 
        whose proportion depend on `percentage`.
    '''
    burnout_distribution = np.ones(n_people)
    burnout_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(burnout_distribution)       
    return burnout_distribution.astype(int)


def generate_pre_existing_addiction_distribution(n_people, percentage):
    ''' Generates proportion of people that 
    experienced addiction in the past year 
    
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    addiction_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose proportions depend on `percentage`.
    '''
    addiction_distribution = np.ones(n_people)
    addiction_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(addiction_distribution)       
    return addiction_distribution.astype(int)


def generate_pre_existing_fatigue_distribution(n_people, percentage):
    ''' Generates proportion of people that 
    experienced chronic fatigue in the past year 
    
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    fatigue_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose 
        proportions depend on `percentage`.
    '''
    fatigue_distribution = np.ones(n_people)
    fatigue_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(fatigue_distribution)       
    return fatigue_distribution.astype(int)


def generate_child_distribution(n_people, percentage):
    ''' Generates proportion of people that had a child since 2013
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    child_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose 
        proportions depend on `percentage`.
    '''
    child_distribution = np.ones(n_people)
    child_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(child_distribution)       
    return child_distribution.astype(int)


def generate_living_with_children_distribution(n_people, percentage):
    ''' Generates proportion of people that currently lives with children
    
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    children_distribution : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose 
        proportions depend on `percentage`.
    '''
    children_distribution = np.ones(n_people)
    children_distribution[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(children_distribution)       
    return children_distribution.astype(int)


def generate_single_parent_distribution(n_people, percentage):
    ''' Generates proportion of people that are single parents
    Parameters
    ----------
    n_people : int
        size of the population.
    percentage : int
        ratio of yes (1) over no (0) (e.g., 50 for 50%)

    Returns
    -------
    single_parent : numpy array
        Array of 1s and 0s (1 = yes, 0 = no) whose proportions depend on `percentage`.
    '''
    single_parent = np.ones(n_people)
    single_parent[:int((n_people * percentage) / 100)] = 0
    np.random.shuffle(single_parent)       
    return single_parent.astype(int)


def generate_health_difficulties_distribution(n_people, no, some, many):
    ''' Generate distribution of people that 
    experienced difficulties with health in the past year
    
    Parameters
    ----------
    n_people : int
        Size of the population.
    no : float
        Probability of not having developed health difficulties in the past year.
    some : float
        Probability of having developed some health difficulties in the past year.
    many : float
        Probability of having developed many health difficulties in the past year. 
        
    Returns
    ----------
    relationship : list
        Distribution of health difficulties in the population.
    '''
    options = {'No': no, 
               'Some': some,
               'Many': many}
    
    choices = list(options.keys())
    weights = list(options.values())
    health = random.choices(choices, weights, k=n_people) 
    return health


def generate_finance_difficulties_distribution(n_people, no, some, many):
    ''' Generate distribution of people that 
    experienced difficulties with finance in the past year
    
    Parameters
    ----------
    n_people : int
        Size of the population.
    no : float
        Probability of not having developed finance difficulties in the past year.
    some : float
        Probability of not having developed some finance difficulties in the past year.
    many : float
        Probability of not having developed many finance difficulties in the past year. 
        
    Returns
    ----------
    relationship : list
        Distribution of finance difficulties in the population.
    '''
    options = {'No': no, 
               'Some': some,
               'Many': many}
    
    choices = list(options.keys())
    weights = list(options.values())
    finance = random.choices(choices, weights, k=n_people) 
    return(finance)


def generate_housing_difficulties_distribution(n_people, no, some, many):
    ''' Generate distribution of people that 
    experienced difficulties with housing in the past year
    
    Parameters
    ----------
    n_people : int
        Size of the population.
    no : float
        Probability of not having developed housing difficulties in the past year.
    some : float
        Probability of not having developed some housing difficulties in the past year.
    many : float
        Probability of not having developed many housing difficulties in the past year. 
        
    Returns
    ----------
    relationship : list
        Distribution of housing difficulties in the population.
    '''
    options = {'No': no, 
               'Some': some,
               'Many': many}
    
    choices = list(options.keys())
    weights = list(options.values())
    housing = random.choices(choices, weights, k=n_people) 
    return housing


def generate_partner_difficulties_distribution(n_people, no, some, many):
    ''' Generate distribution of people that 
    experienced difficulties with their partner in the past year
    
    Parameters
    ----------
    n_people : int
        Size of the population.
    no : float
        Probability of not having developed partner difficulties in the past year.
    some : float
        Probability of not having developed some partner difficulties in the past year.
    many : float
        Probability of not having developed many partner difficulties in the past year. 
        
    Returns
    ----------
    relationship : list
        Distribution of partner difficulties in the population.
    '''
    options = {'No': no, 
               'Some': some,
               'Many': many}
   
    choices = list(options.keys())
    weights = list(options.values())
    partner = random.choices(choices, weights, k=n_people) 
    return partner

    
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
    gd = generate_gender_distribution(n_people=n_people, percentage=parameters['%_gender'])
    ad = generate_age_distribution(n_people=n_people, n_age_groups=parameters['n_age_groups'], min_age=parameters['min_age'], max_age=parameters['max_age'], prob=parameters['age_strata'])
    ea = generate_educational_attainment_distribution(n_people=n_people, low=parameters['low_education'], medium=parameters['medium_education'], high=parameters['high_education'])
    ed = generate_employment_distribution(n_people=n_people, yes=parameters['employed_true'], no_seeking=parameters['employed_false_seeking'], no_other=parameters['employed_false_other'])
    pd = generate_partnership_statuts_distribution(n_people=n_people, single=parameters['single'], married=parameters['married'], live_in_partner=parameters['live_in_partner'], in_relationship_no_cohabitation=parameters['in_relationship_no_cohabitation'], other=parameters['other_relationship'])
    dd = generate_pre_existing_depression_distribution(n_people=n_people, percentage=parameters['%_depression'])
    pad = generate_pre_existing_addiction_distribution(n_people=n_people, percentage=parameters['%_addiction'])
    pfd = generate_pre_existing_fatigue_distribution(n_people=n_people, percentage=parameters['%_fatigue'])
    cd = generate_child_distribution(n_people=n_people, percentage=parameters['%_child'])
    pbd = generate_pre_existing_burnout_distribution(n_people=n_people, percentage=parameters['%_burnout'])
    lcd = generate_living_with_children_distribution(n_people=n_people, percentage=parameters['%_child_inhouse'])
    hd = generate_health_difficulties_distribution(n_people=n_people, no=parameters['health_difficulties_false'], some=parameters['health_difficulties_some'], many=parameters['health_difficulties_many'])
    fd = generate_finance_difficulties_distribution(n_people=n_people, no=parameters['finance_difficulties_false'], some=parameters['finance_difficulties_some'], many=parameters['finance_difficulties_many'])
    spd = generate_single_parent_distribution(n_people=n_people, percentage=parameters['%_single_parent'])
    hdd = generate_housing_difficulties_distribution(n_people=n_people, no=parameters['housing_difficulties_false'], some=parameters['housing_difficulties_some'], many=parameters['housing_difficulties_many'])
    pdd = generate_partner_difficulties_distribution(n_people=n_people, no=parameters['partner_difficulties_false'], some=parameters['partner_difficulties_some'], many=parameters['partner_difficulties_many'])
    people = []

    for i in range(n_people):
        features = Features(name=str(i),
                            gender=gd[i], 
                            age=ad[i], 
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
                            partner_difficulties=pdd[i])
        
        state = State(employed=np.random.choice(2),
                      infected=0,
                      mental_health=2,
                      n_contacts=0)
        
        people.append(Agent(features=features, state=state))
    return people

def make_features_dataframe(agent_list):
    '''
    Takes a list of agents, 
    returns pandas dataframe of features.
    
    Parameters
    ----------
    agent_list : list
        List of dataclasses of the type Agent

    Returns
    -------
    df : pandas dataframe
        Dataframe of the agents
    '''
    df = []
    for p in agent_list:
        df.append(asdict(p.features))
    df = pd.DataFrame.from_dict(df)
    return df


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
    dataframe = make_features_dataframe(agent_list=agent_list)
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
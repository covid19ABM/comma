"""
Functions to generate ad-hoc feature distributions.
"""

import numpy as np
import random 

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

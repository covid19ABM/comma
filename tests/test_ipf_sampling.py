from comma.individual import Individual
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

# This test is linked to issue #53 https://github.com/covid19ABM/comma/issues/53
# The test's aim is to check that the distribution generated is
# matching the marginals of the LA's orginal distribution (provided as cross tabs).

gender_names = ['m', 'f']
age_group_names = ["_1", "_2", "_3", "_4"] # "24-34", "35-44", "45-54", "55-64"
education_names = ["low", "middle", "high"]
unemployed_names = ["yes", "no"]
have_partner_names = ["yes", "no"]
depressed_names = ["yes", "no"]
children_presence_names = ["yes", "no"]
housing_financial_difficulties_names = ["yes", "no"]
selfrated_health_names = ["good", "average", "poor"]
critical_job_names = ["yes", "no"]

# define the cross table gender x education_level
gender = pd.DataFrame([[86, 238, 725], [106, 396, 1104]], index=gender_names, columns=education_names)
# define the cross table age_group x education_level
age_group = pd.DataFrame([[6, 97, 539], [34, 161, 486], [109, 296, 596], [43, 80, 208]], index=age_group_names, columns=education_names)
# define the cross table education_level x unemployed
education = pd.DataFrame([[10, 182], [31, 603], [65, 1764]], index=education_names, columns=unemployed_names)
# Define the dataframe for Partner x Depressed
have_partner = pd.DataFrame([[45, 2069], [21, 520]], index=have_partner_names, columns=depressed_names)
# Define the dataframe for Depressed x Children
depressed = pd.DataFrame([[27, 39], [1310, 1279]], index=depressed_names, columns=children_presence_names)
# Define the dataframe for Children x Housing/Financial troubles
children_presence = pd.DataFrame([[292, 1045], [476, 842]], index=children_presence_names, columns=housing_financial_difficulties_names)
# Define the dataframe for Unemployed x Partner
unemployed = pd.DataFrame([[70, 36], [2044, 505]], index=unemployed_names, columns=have_partner_names)
# Define the dataframe for Housing/Financial trouble x Self-rated health
housing_financial_difficulties = pd.DataFrame([[874, 928, 85], [269, 436, 63]], index=housing_financial_difficulties_names, columns=selfrated_health_names)
# Define the dataframe for Housing/Financial trouble x Critical job
critical_job = pd.DataFrame([[304, 464], [808, 1079]], index=housing_financial_difficulties_names, columns=critical_job_names)
# Define the dataframe for Self-rated health x Critical job
selfrated_health = pd.DataFrame([[471, 672], [571, 793], [70, 78]], index=selfrated_health_names, columns=critical_job_names)


def chisq_of_df_cols(df, c1, c2):
    """
    Conduct a Chi-squared test of independence on two categorical columns of a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        c1 (str): The name of the first column to test.
        c2 (str): The name of the second column to test.

    Returns:
        float: The p-value from the Chi-squared test.
    """
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which may cause exceptions
    result = chi2_contingency(ctsum.fillna(0))
    return result[1] # return only the p-value

@pytest.mark.parametrize('time', range(100))
def test_data_sampling_ipf(time):
    """Test if the sampling result aligns to the cross-tabs.
    """
    
    size = 2655 # This is the original sample size provided by Kristina
    dir_params = "./parameters"
    sample_set = Individual.sampling_from_ipf(size, dir_params)
    cols = sample_set.columns.tolist()
    
    # Define crosstabs of the original sample (i.e., truth). 
    # We have 43 crosstabs in total, however, they are ad-hoc combinations of 9 variables.
    # It's sufficient to test whether the distribution of those 9 variables changes
    # significantly than the original sample, rather than all 43 combinations.
    
    crosstabs_dict = {
    'original_gender': gender,
    'original_age_group': age_group,
    'original_education_level': education,
    'original_partner': have_partner,
    'original_depressed': depressed,
    'original_children': children_presence,
    'original_unemployed': unemployed,
    'original_housing_financial_trouble': housing_financial_difficulties,
    'original_selfrated_health': selfrated_health,
    'original_critical_job': critical_job
    }

    # df that stores the results
    results_df = pd.DataFrame(columns=['var', 'key', 'pvalue'])
    
    for var, key in zip(cols, crosstabs_dict):
        # group by variable and count the instances
        cross_ipf = sample_set.groupby(var).count().iloc[:, 0]
        # create a DataFrame with index as the variable type and count as the number of instances
        df_cross_ipf = pd.DataFrame({'type': cross_ipf.index, 'count': cross_ipf.values})
        # add a 'sample' column to each df before concatenating
        df_cross_ipf['sample'] = 'ipf'
        
        # get the corresponding crosstab from the original sample of Kristina
        crosstab = crosstabs_dict[key]
        cross_ground_truth = crosstab.sum(axis=1)
        # format the df as per above
        df_cross_ground_truth = pd.DataFrame({'type': cross_ground_truth.index, 'count': cross_ground_truth.values})
        df_cross_ground_truth['sample'] = 'truth'
        # Reindex the ipf DataFrame based on the truth DataFrame (sometimes the order of the variables is inverted)
        df_cross_ipf.set_index('type', inplace=True)
        df_cross_ipf = df_cross_ipf.reindex(df_cross_ground_truth['type'])
        df_cross_ipf.reset_index(inplace=True)
        
        # put together the dataframes
        df_combined = pd.concat([df_cross_ipf, df_cross_ground_truth])

        # Calculate the p-value from the Chi-squared test
        p_value = chisq_of_df_cols(df_combined, 'type', 'sample')

        # store the results
        result = pd.DataFrame({'var': [var], 'key': [key], 'pvalue': [p_value]})
        results_df = pd.concat([results_df, result], ignore_index=True)

    # return anything that is significant
    assert all(results_df['pvalue'] > 0.05)
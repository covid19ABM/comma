from comma.individual import Individual
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

# Test linked to issue #53 https://github.com/covid19ABM/comma/issues/53
# The test's aim is to check that the distribution generated is
# matching the marginals of the LA's
# original distribution (provided as cross tabs).


class TestDataSamplingIPF:
    # Define crosstabs of the original sample (i.e., truth).
    # We have >43 crosstabs in total, however,
    # they are ad-hoc combinations of 13 variables.
    # It's sufficient to test whether the distribution
    # of those 13 variables changes significantly than the
    # original sample, rather than all combinations.
    @pytest.fixture(scope="class")
    def gender_names(self):
        return ["m", "f"]

    @pytest.fixture(scope="class")
    def age_group_names(self):
        return ["_1", "_2", "_3", "_4"]

    @pytest.fixture(scope="class")
    def education_names(self):
        return ["low", "middle", "high", "unknown"]

    @pytest.fixture(scope="class")
    def unemployed_names(self):
        return ["yes", "no"]

    @pytest.fixture(scope="class")
    def have_partner_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def depressed_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def children_presence_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def housing_financial_difficulties_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def selfrated_health_names(self):
        return ["good", "average", "poor", "unknown"]

    @pytest.fixture(scope="class")
    def critical_job_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def BMI_names(self):
        return ["underweight", "normalweight", "overweight", "obese", "unknown"]

    @pytest.fixture(scope="class")
    def living_alone_names(self):
        return ["yes", "no", "unknown"]

    @pytest.fixture(scope="class")
    def income_median_names(self):
        return ["below", "above", "unknown"]

    # define the cross table gender x education_level
    @pytest.fixture(scope="class")
    def gender(self, gender_names, education_names):
        return pd.DataFrame(
            [[170, 382, 1047, 917], [269, 616, 1596, 916]],
            index=gender_names,
            columns=education_names,
        )

    # define the cross table age_group x education_level
    @pytest.fixture(scope="class")
    def age_group(self, age_group_names, education_names):
        return pd.DataFrame(
            [
                [9, 156, 774, 538],
                [57, 236, 655, 332],
                [199, 409, 789, 431],
                [174, 197, 425, 532],
            ],
            index=age_group_names,
            columns=education_names,
        )

    # define the cross table education_level x unemployed
    @pytest.fixture(scope="class")
    def education(self, education_names, unemployed_names):
        return pd.DataFrame(
            [[36, 403], [67, 931], [118, 2525], [49, 1784]],
            index=education_names,
            columns=unemployed_names,
        )

    # Define the dataframe for Partner x Depressed
    @pytest.fixture(scope="class")
    def have_partner(self, have_partner_names, depressed_names):
        return pd.DataFrame(
            [[126, 3689, 78], [65, 921, 24], [28, 966, 16]],
            index=have_partner_names,
            columns=depressed_names,
        )

    # Define the dataframe for Depressed x Children
    @pytest.fixture(scope="class")
    def depressed(self, depressed_names, critical_job_names):
        return pd.DataFrame(
            [[742, 958, 274], [869, 1255, 860], [334, 377, 244]],
            index=depressed_names,
            columns=critical_job_names,
        )

    # Define the dataframe for Children x Housing/Financial troubles
    @pytest.fixture(scope="class")
    def children_presence(
        self, children_presence_names, housing_financial_difficulties_names
    ):
        return pd.DataFrame(
            [[450, 1347, 177], [835, 1922, 227], [269, 469, 217]],
            index=children_presence_names,
            columns=housing_financial_difficulties_names,
        )

    # Define the dataframe for Unemployed x Partner
    @pytest.fixture(scope="class")
    def unemployed(self, unemployed_names, have_partner_names):
        return pd.DataFrame(
            [[183, 85, 2], [3710, 925, 1008]],
            index=unemployed_names,
            columns=have_partner_names,
        )

    # Define the dataframe for Housing/Financial trouble x Self-rated health
    @pytest.fixture(scope="class")
    def housing_financial_difficulties(
        self, housing_financial_difficulties_names, selfrated_health_names
    ):
        return pd.DataFrame(
            [[516, 860, 224, 6], [1605, 1902, 170, 6], [6, 15, 6, 597]],
            index=housing_financial_difficulties_names,
            columns=selfrated_health_names,
        )

    # Define the dataframe for Housing/Financial trouble x Critical job
    @pytest.fixture(scope="class")
    def critical_job(self, housing_financial_difficulties_names, critical_job_names):
        return pd.DataFrame(
            [[508, 695, 351], [1260, 1629, 849], [177, 266, 178]],
            index=housing_financial_difficulties_names,
            columns=critical_job_names,
        )

    # Define the dataframe for Self-rated health x Critical job
    @pytest.fixture(scope="class")
    def selfrated_health(self, selfrated_health_names, critical_job_names):
        return pd.DataFrame(
            [[756, 999, 373], [917, 1211, 649], [95, 119, 183], [177, 261, 173]],
            index=selfrated_health_names,
            columns=critical_job_names,
        )

    @pytest.fixture(scope="class")
    def BMI(self, BMI_names, gender_names):
        return pd.DataFrame(
            [[4, 71], [1169, 2113], [893, 972], [256, 427], [4, 4]],
            index=BMI_names,
            columns=gender_names,
        )

    @pytest.fixture(scope="class")
    def living_alone(self, living_alone_names, depressed_names):
        return pd.DataFrame(
            [[3613, 244, 33], [280, 766, 22], [319, 318, 318]],
            index=living_alone_names,
            columns=depressed_names,
        )

    @pytest.fixture(scope="class")
    def income_median(self, income_median_names, education_names):
        return pd.DataFrame(
            [[175, 306, 560, 291], [77, 253, 1062, 337], [187, 439, 1021, 1205]],
            index=income_median_names,
            columns=education_names,
        )

    @pytest.fixture(scope="class")
    def crosstabs_dict(
        self,
        gender,
        age_group,
        education,
        have_partner,
        depressed,
        children_presence,
        unemployed,
        housing_financial_difficulties,
        selfrated_health,
        critical_job,
        BMI,
        living_alone,
        income_median,
    ):
        return {
            "original_gender": gender,
            "original_age_group": age_group,
            "original_education_level": education,
            "original_partner": have_partner,
            "original_depressed": depressed,
            "original_children": children_presence,
            "original_unemployed": unemployed,
            "original_housing_financial_trouble": housing_financial_difficulties,
            "original_selfrated_health": selfrated_health,
            "original_critical_job": critical_job,
            "original_bmi": BMI,
            "original_living_alone": living_alone,
            "original_income_median": income_median,
        }

    @staticmethod
    def chisq_of_df_cols(df, c1, c2):
        """
        Conduct a Chi-squared test of independence
        on two categorical columns of a DataFrame.

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
        return result[1]  # return only the p-value

    def test_data_sampling_ipf(self, crosstabs_dict):
        """Test if the sampling result aligns to the cross-tabs"""
        size = 5913  # This is the original sample size provided by Kristina
        dir_params = "./parameters"
        sample_set = Individual.sampling_from_ipf(size, dir_params)
        cols = sample_set.columns.tolist()

        # df that stores the results
        results_df = pd.DataFrame(columns=["var", "key", "pvalue"])

        for var, key in zip(cols, crosstabs_dict):
            cross_ipf = sample_set.groupby(var).count().iloc[:, 0]
            df_cross_ipf = pd.DataFrame(
                {"type": cross_ipf.index, "count": cross_ipf.values}
            )
            df_cross_ipf["sample"] = "ipf"

            crosstab = crosstabs_dict[key]
            cross_ground_truth = crosstab.sum(axis=1)
            df_cross_ground_truth = pd.DataFrame(
                {"type": cross_ground_truth.index, "count": cross_ground_truth.values}
            )
            df_cross_ground_truth["sample"] = "truth"
            df_cross_ipf.set_index("type", inplace=True)
            df_cross_ipf = df_cross_ipf.reindex(df_cross_ground_truth["type"])
            df_cross_ipf.reset_index(inplace=True)

            # put together the dataframes
            df_combined = pd.concat([df_cross_ipf, df_cross_ground_truth])

            # Calculate the p-value from the Chi-squared test
            p_value = TestDataSamplingIPF.chisq_of_df_cols(
                df_combined, "type", "sample"
            )

            # store the results
            result = pd.DataFrame({"var": [var], "key": [key], "pvalue": [p_value]})
            results_df = pd.concat([results_df, result], ignore_index=True)
        assert all(results_df["pvalue"] > 0.05)

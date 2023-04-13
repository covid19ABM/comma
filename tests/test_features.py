""" A test suite for the Feature data class """

from contextlib import redirect_stdout
from dataclasses import fields
import io
import pytest
from mhm.agent import Features


@pytest.fixture
def feat():
    """ Initialises a Feature object to use in the tests. """   
    return Features(name=1, gender=1, age=30, age_group=1, education='Low',
                    employed='Yes', partnership_status='In relationship, no cohabitation',
                    pre_existing_depression=0, pre_existing_burnout=1,
                    pre_existing_addiction=0, pre_existing_chronic_fatigue=0,
                    parenthood=1, living_with_child=1, single_parent=0,
                    housing_difficulties='Some', finance_difficulties='Many',
                    pre_existing_health_issues='No', partner_difficulties='No',
                    job_type='key_worker')


def summary_to_dict(obj):
    """ Converts the output of the summary() method of an object into a dict,
    with string values for all fields except for integer values."""  
    output = io.StringIO()
    with redirect_stdout(output):
        obj.summary()
    lines = output.getvalue().strip().split('\n')
    result = {}
    for line in lines:
        key, val = line.split(':')
        key = key.strip()
        val = val.strip()
        if val.isdigit():
            result[key] = int(val)
        else:
            result[key] = str(val)
    return result


def test_summary(feat):
    """ Tests the summary method of the Features class.
    Verifies that the output of the summary method matches the expected output """
    expected_output = {
        'name': 1,
        'gender': 1,
        'age': 30,
        'age_group': 1,
        'education': 'Low',
        'employed': 'Yes',
        'partnership_status': 'In relationship, no cohabitation',
        'pre_existing_depression': 0,
        'pre_existing_burnout': 1,
        'pre_existing_addiction': 0,
        'pre_existing_chronic_fatigue': 0,
        'parenthood': 1,
        'living_with_child': 1,
        'single_parent': 0,
        'housing_difficulties': 'Some',
        'finance_difficulties': 'Many',
        'pre_existing_health_issues': 'No',
        'partner_difficulties': 'No',
        'job_type': 'key_worker',
        }
    output = summary_to_dict(feat)
    assert output == expected_output


def test_fields(feat):
    """ Tests the fields of the Features class. 
    Verifies that the names of the fields in the Features class match the expected list of field names. """
    expected_fields = [
        'name', 'gender', 'age', 'age_group', 'education', 'employed', 'partnership_status',
        'pre_existing_depression', 'pre_existing_burnout', 'pre_existing_addiction',
        'pre_existing_chronic_fatigue', 'parenthood', 'living_with_child', 'single_parent',
        'housing_difficulties', 'finance_difficulties', 'pre_existing_health_issues',
        'partner_difficulties', 'job_type'
    ]
    actual_fields = [f.name for f in fields(feat)]
    assert actual_fields == expected_fields
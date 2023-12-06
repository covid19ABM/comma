All notable changes to the codebase are documented in this file.

.. contents:: **Contents**
   :local:
   :depth: 1

~~~~~~~~~
1.1.2
~~~~~~~~~
Released on December 6, 2023

Main changes
-------------

Features
^^^^^^^^

In `PR #76 <https://github.com/covid19ABM/comma/pull/76>`_ by @n400peanuts:

- Integration of the long covid type of situation in the individual class
- Addition of the `get_covid_status` among the methods in the individual class
- `is_recovered()` method now has been adjusted to integrate long-covid type of recovery
- Change of the data repository to the RIVM website (cumulative positive tests) in the hypothesis class
- Issue about having an additional method to work out positive cases for days that are not in data -- solved, this is integrated into the RIVM data already
- Hypothesis class does not require anymore the following methods: `get_file_paths()` and `filter_dates`
- Hypothesis class now has adjusted methods for `get_covid_data()` and `get_positive_cases()` to account for the RIVM cumulative tests data
- Model class now accounts for municipality_code rather than 'location'

Tests
^^^^^

In `PR #76 <https://github.com/covid19ABM/comma/pull/76>`_ by @n400peanuts:

- Added test for long covid with `test_long_covid`
- Added test for testing positivity to COVID with `test_positivity`
- tests `test_filter_dates_on_boundary`, `test_filter_dates_within_range` and `file_list` have all been removed, as the new data collection from RIVM does not require anymore the functions `filter_dates`
- tests `test_range_error` and `test_get_covid_data` have been changed to include the new RIVM website and type of data
- tests `test_scaling` and `test_location_not_in_dataset` changed to include the new RIVM website and type of data

~~~~~~~~~
1.1.1
~~~~~~~~~

Released on November 21, 2023

Main changes
-------------

Refactor
^^^^^^^^

- refactor: hypothesis class focuses now on data validation and preparation for the model `PR #75 <https://github.com/covid19ABM/comma/pull/75>`_ by @n400peanuts
- refactor: model class now requires only initialisation and run, no more setup() `PR #75 <https://github.com/covid19ABM/comma/pull/75>`_ by @n400peanuts
- refactor: individual class better interacts with the model and hypothesis class `PR #75 <https://github.com/covid19ABM/comma/pull/75>`_ by @n400peanuts

~~~~~~~~~
1.1.0
~~~~~~~~~

Released on October 18, 2023

Main changes
-------------

Features
^^^^^^^^
- Comma now integrates positivity to COVID-19 in the model `Commit [2ff9a2c] in PR #73 <https://github.com/covid19ABM/comma/pull/73/commits/2ff9a2c736a8b2a9c9235cea6a4c8d090c7d27dd>`_ by @n400peanuts
- New documentation about the implementation of the positivity to COVID-19 `PR #48 <https://github.com/covid19ABM/comma/commit/37372a3c46202d650297a285f091810914caddb1>`_ by @n400peanuts
- Models can be set with a specific seed to ensure reproducibility  `Commit [3a9419] in PR #73 <https://github.com/covid19ABM/comma/pull/73/commits/3a9419446e502b50e8cc667e4ff9737ea622e871>`_ by @n400peanuts
- Code formatted with Black

Tests
^^^^^
- Added tests for Hypothesis class `Commit [5636c9] in PR #73 <https://github.com/covid19ABM/comma/commit/5636c9e6221da6d14ca9662a7947cbcda2d51ebc>`_ by @n400peanuts
- Added test for Individual "test_actions_when_positive" `Commit [] in PR #73 <https://github.com/covid19ABM/comma/commit/5636c9e6221da6d14ca9662a7947cbcda2d51ebc>`_ by @n400peanuts
- Refactored test for Model module to have higher tolerance and test all data types in dataframe `Commit [8e007] in PR #73 <https://github.com/covid19ABM/comma/pull/73/commits/8e007980e8cbc43d2db0fe49c2b86cc256205839>`_ by @n400peanuts


~~~~~~~~~
1.0.0
~~~~~~~~~

Released on August 29, 2023

Main changes
-------------


Refactor
^^^^^^^^

- refactor: generate agents with the IPF sampling and refactor individual class in `PR #60 <https://github.com/covid19ABM/comma/pull/60>`_ by @n400peanuts
- refactor: generate agents by sampling from the param_individual.json in `PR #22 <https://github.com/covid19ABM/comma/pull/22>`_ by @jiqicn
- refactor: create model class in `PR #18 <https://github.com/covid19ABM/comma/pull/18>`_ by @n400peanuts
- refactor: create individual class in `PR #17 <https://github.com/covid19ABM/comma/pull/17>`_ by @jiqicn
- refactor: add new actions effects on mental health matrices in `PR #72 <https://github.com/covid19ABM/comma/pull/72>`_ by @Astrid-p

Features
^^^^^^^^
- feat: add tutorial notebook in `PR #43 <https://github.com/covid19ABM/comma/pull/43>`_ by @n400peanuts

Tests
^^^^^
- feat: add tests for IPF sampling in `PR #59 <https://github.com/covid19ABM/comma/pull/59>`_ by @n400peanuts
- feat: add tests for checking matrices in `PR #32 <https://github.com/covid19ABM/comma/pull/32>`_ by @n400peanuts and @jiqicn
- feat: add test suite in `PR #71 <https://github.com/covid19ABM/comma/pull/71>`_ by @n400peanuts
- feat: add test for `take_actions` in `PR #67 <https://github.com/covid19ABM/comma/pull/67>`_ by @n400peanuts

Docs
^^^^
- docs: docs update in `PR #31 <https://github.com/covid19ABM/comma/pull/31>`_ by n400peanuts
- docs: add documentation in `PR #44 <https://github.com/covid19ABM/comma/pull/44>`_ by n400peanuts

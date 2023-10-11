========================================
How agents become positive for COVID-19?
========================================


In this document, we will consider the dynamics of how agents become positive for COVID-19 in our model.

Before that, a foreword:

The coronavirus pandemic from 2020 to 2023 made almost everyone globally realise the importance of
understanding contagion processes. This led to a surge in modelling work highlighting contagion mechanisms.
*Comma*, however, does not focus on this aspect; rather it addresses the mental health consequences of *testing*
positive for COVID-19.

This was a deliberate decision. We believe that there's ample information out there that elucidated
how COVID-19 spreads, and we aren't keen to further explore that area. More crucially, with hindsight, we now understand how the virus spread within specific populations during
particular periods. This means we aren't limited to hypothetical models; we can exploit real data from specific
time points, fully aware of the stage of the COVID-19 virus's spread at those moments.

A brief caveat to bear in mind:
The statistics reflect only the recorded cases resulting from COVID-19 tests (any tests).
A dip in testing numbers doesn't necessarily indicate diminished contagion.
Nonetheless, these are the metrics that governments use to implement lockdowns policies.
Lockdown – operationalised as restrictions on actions – is our primary independent variable.
As a consequence, this is the lens through which we examine how a positive COVID-19 diagnosis might impact mental health in our simulation.

With that said, let's delve into how Comma has approached the impact of COVID-19 positivity.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 1: Compute new COVID-19 positive cases in the population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We extract data from the `RIVM github repository <https://github.com/mzelst/covid-19/raw/master/data-rivm/tests/>`_.
We select a period of time, and a location and we compute the number of new cases in the simulated population
as:

.. math:: cases = (n/N)*M

where :math:`N` is te size of the real population for that specific location,
:math:`M` is the size of the simulated population, :math:`n` is the number of new positives reported on RIVM.

In the image below, we have selected 90 days from '2022-01-01' to '2022-03-30' in
Groningen. We have simulated a population of 10000 individuals and we plotted how many people
per day become positive, and conversely, how many stay negative.

[|image1|]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step 2: Set how long agents will be infectious / definition of recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decourse must have a distribution with a peak ~ 10 days, and have a very long tail (this decision is based on `literature <https://www.nature.com/articles/d41586-022-02026-x#ref-CR1>`_).
Given these premises a good approximation could be a gamma distribution with :math:`a`=5 and :math:`scale`=3.
In addition, around `20% <https://www.rivm.nl/en/coronavirus-covid-19/long-covid>`_ of the population infected will stay positive for longer than 35 days (Long COVID).

[|image2|]

Agents will also recover (i.e., their status will switch from positive to negative) following the gamma distribution.
Note that whenever their status is set to Negative, agents can be potentially re-infected.


.. |image1| image:: https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/new_covid_cases.png
.. |image2| image:: https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/gamma.png


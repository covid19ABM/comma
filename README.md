# COMMA - **CO**vid **M**ental-health **M**odel with **A**gents 

<div align="center">
<img src="docs/img/avatar_comma.png">
</div>

`comma` lets you run a agent-based simulations to study mental health outcomes during covid-19 lockdowns. 

# Why?
This project aims at understanding the full spectrum of impacts the lockdown policies had during the COVID-19 pandemic, specifically on non-COVID-19-related health outcomes, such as mental health. Although lockdowns reduced disease transmission and mortality, they also potentially exacerbated mental health issues. By using `comma` you can simulate real-world scenarios, and estimate/compare the effects of lockdown policies on the mental health of an a-priori defined population across time.

# Project status[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#project-status)
[![Python package](https://github.com/covid19ABM/comma/actions/workflows/python-package.yml/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/python-package.yml) 
[![pages-build-deployment](https://github.com/covid19ABM/comma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/pages/pages-build-deployment)
[![Docs](https://github.com/covid19ABM/comma/actions/workflows/documentation.yaml/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/documentation.yaml)

# Installation

## Prerequisites
- Python 3.6 or above 
- [Poetry (Python packaging and dependency management tool)](https://python-poetry.org/docs/#installation)

## Install from source

```bash
git clone git@github.com:covid19ABM/comma.git
cd comma
poetry install
```

After the package is installed, you can activate the virtual environment to use the package:
```bash
poetry shell
```

That's it! After following these steps, you should have `comma` installed in a dedicated virtual environment and be ready to use it.

# Credits
This is a project funded by the Netherlands eScience Center (Grant ID: NLESC.SSI.2022b.022) and awarded to Dr Kristina Thompson (Wageningen University) and developed in collaboration with the Netherlands eScience Center. More information on the [Research Software Directory](https://research-software-directory.org/projects/covid-19-mitigation-policies).

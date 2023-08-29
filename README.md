# COMMA - **CO**vid **M**ental-health **M**odel with **A**gents 

<div align="center">
<img src="docs/img/avatar_comma.png">
</div>

`comma` lets you run agent-based simulations to study mental health outcomes during covid-19 lockdowns.

# Project status[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#project-status)

[![Python package](https://github.com/covid19ABM/comma/actions/workflows/python-package.yml/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/python-package.yml) 
[![pages-build-deployment](https://github.com/covid19ABM/comma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/pages/pages-build-deployment)
[![Docs](https://github.com/covid19ABM/comma/actions/workflows/documentation.yaml/badge.svg)](https://github.com/covid19ABM/comma/actions/workflows/documentation.yaml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=covid19ABM_comma&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=covid19ABM_comma)
[![github license badge](https://img.shields.io/github/license/covid19ABM/comma)](https://github.com/covid19ABM/comma)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7777/badge)](https://www.bestpractices.dev/projects/7777)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=covid19ABM_comma&metric=coverage)](https://sonarcloud.io/summary/new_code?id=covid19ABM_comma)

# Why?[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#why)
This project aims at understanding the full spectrum of impacts the lockdown policies had during the COVID-19 pandemic, specifically on non-COVID-19-related health outcomes, such as mental health. Although lockdowns reduced disease transmission and mortality, they also potentially exacerbated mental health issues. By using `comma` you can simulate real-world scenarios, and estimate/compare the effects of lockdown policies on the mental health of an a-priori defined population across time.

# Table of contents[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#table-of-contents)
- [Motivation](#why)
- [Installation](#installation)
- [Example](#example)
- [Diagram](#diagram)
- [License](#license)
- [Credits](#credits)

# Installation[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#installation)

## Prerequisites
- Python 3.6 or above 

## Install from source

We recommend installing `comma` in a virtual environment. For example, in conda:
```bash
conda create --name comma_env
conda activate comma_env
```

Then installing `comma` with `pip` by cloning the github repository locally:
```bash
git clone git@github.com:covid19ABM/comma.git
cd comma
python -m pip install .

```

That's it! After following these steps, you should have `comma` installed in a dedicated virtual environment and be ready to use it.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

# Example[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#example)
You can find a tutorial that demonstrates the usage of `comma` in the `/notebooks` folder.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

# License[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#license)
`comma` is under free open source [Apache License Version 2.0](https://raw.githubusercontent.com/covid19ABM/comma/main/LICENSE). This means that you're free to use, modify, and distribute this software, even for commercial applications.

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

# Diagram[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#diagram)
![diagram](https://github.com/covid19ABM/comma/blob/main/comma_diagram.drawio.svg)

<div align="right">[ <a href="#table-of-contents">↑ Back to top ↑</a> ]</div>

# Credits[![](https://raw.githubusercontent.com/covid19ABM/comma/main/docs/img/pin.svg)](#credits)
This is a project funded by the Netherlands eScience Center (Grant ID: NLESC.SSI.2022b.022) and awarded to Dr Kristina Thompson (Wageningen University) and developed in collaboration with the Netherlands eScience Center. More information on the [Research Software Directory](https://research-software-directory.org/projects/covid-19-mitigation-policies).


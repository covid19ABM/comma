from setuptools import find_packages, setup


with open('README.md', 'r') as file:
    LONG_DESC = file.read()

INSTALL_REQUIRES = ['pandas', 'matplotlib']
EXTRA_REQUIRES = ['pytest', 'jupyter']


setup(
    name="mhm",
    version='0.1.0',
    description="An agent-based simulation model to study mental health outcomes during covid-19 lockdowns.",
    long_description=LONG_DESC, 
    long_description_content_type='text/markdown', 
    url="https://github.com/covid19ABM/mhm",
    author="Eva Viviani, Ji Qi",
    packages=find_packages(exclude=["tests", ".github"]),
    python_requires='>=3.6', 
    install_requires=INSTALL_REQUIRES,
    extras_require={"test": EXTRA_REQUIRES},
)
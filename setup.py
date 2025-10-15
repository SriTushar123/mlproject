'''
This script is created to make my ML application as a package
'''
from setuptools import setup,find_packages
hypen_e="-e ."
def get_requirements(path):
    " This funtion will return the list of requirements "
    requirements=[]
    with open(path) as file:
        requirements=file.readlines()
    
    requirements=[i.replace("\n","") for i in requirements]
    if (hypen_e in requirements):
        requirements.remove(hypen_e)
    
    return requirements

setup(
name="mlproject",
version="0.0.1",
author="Tushar",
author_email="tusharsrivastava354@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)
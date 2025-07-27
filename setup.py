from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        requirements = [line.strip() for line in f.readlines() if line.strip()]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='ml_project_template',
    version='0.0.1',
    author='Bharat',
    author_email='bharatsinghal209@example.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    )
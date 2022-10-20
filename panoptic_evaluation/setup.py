from setuptools import setup

setup(
   name='panoptic_evaluation',
   version='1.0',
   description='Module to evaluate OOD',
   author='Kiran',
   author_email='foomail@foo.example',
   packages=['panoptic_evaluation'],
   install_requires=['wheel', 'cityscapesscripts'], #external packages as dependencies
)
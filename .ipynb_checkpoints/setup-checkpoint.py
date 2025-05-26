from setuptools import setup, find_packages

setup(
    name='cv_unstable_correlations',
    version='0.1',
    description='CV approach that results in models that have higher and more stable performance than existing CV approaches.',
    author='Meera Krishnamoorthy',
    packages=find_packages(),
    install_requires=['numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
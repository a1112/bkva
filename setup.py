from setuptools import setup, find_packages, Extension
import glob

setup(
    name='BKVisionalgorithms',
    version='0.0.2',
    description='BKVisionalgorithms',
    author='BKVision',
    author_email='',
    license="MIT Licence",
    packages=find_packages(exclude=('BKVisionalgorithms', 'utils')),
    platforms="any",
    package_data={
    },
    # package_dir={'': "NerCarDataBase"},
    # package_data = {
    #  '': ["*.py"],
    #  'NerCarDataBase': ["*.*"]
    # },
    # url='',
    # py_modules=[py.split('.')[0] for py in glob.glob("NerCarDataBase/*.py")],
    install_requires=[],
    zip_safe=False
)

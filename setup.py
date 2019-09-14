from setuptools import find_packages, setup


setup(
    name='chainer_nn',
    version='0.1.0',
    author='Hiroki Teranishi',
    author_email='teranishihiroki@gmail.com',
    description='chainer neural network implementation',
    url='https://github.com/chantera/chainer_nn',
    install_requires=['chainer>=2.0.0'],
    packages=find_packages(),
)

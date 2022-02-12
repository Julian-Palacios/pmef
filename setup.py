from setuptools import setup

setup(
    name='pmef',
    url='https://github.com/Julian-Palacios/pmef',
    author='Julian Palacios, Jorge Lulimachi, Humberto Rojas, Luis Baldeon',
    author_email='jpalaciose@uni.pe',
    # Needed to actually package something
    packages=['pmef'],
    # Needed for dependencies
    install_requires=['numpy','scipy'],
    version='0.1',
    license='MIT',
    description='Codigo para aprender a programar el Metodo de ELementos Finitos',
    keywords=['Elementos Finitos'],
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
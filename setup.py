from setuptools import setup

setup(
    name='pmef',
    url='https://github.com/Julian-Palacios/pmef',
    author='Julian Palacios, Jorge Lulimachi, Luis Baldeon y Humberto Rojas',
    author_email='jpalaciose@uni.pe',
    # Needed to actually package something
    packages=['pmef'],
    # Needed for dependencies
    install_requires=['numpy','scipy','matplotlib','meshio'],
    version='0.5.3',
    license='GPL-3.0',
    description='Librería para aprender a Programar el Método de Elementos Finitos',
    keywords=['Elementos Finitos','Funciones de Forma','Mapeo Isoparamétrico'],
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
# Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe)
# Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
# Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##
# Este código se basó en las rutinas de matlab FEMCode
# realizado inicialmente por Garth N .Wells (2005)
# para la clase CT5123 en TU Delft, Países Bajos
##

import matplotlib.tri as tri
import matplotlib.colors as colors
import matplotlib.animation as animation

import numpy as np
from numpy.lib.shape_base import kron
from numpy.linalg import det, inv
from scipy import sparse
import matplotlib.pyplot as plt
import pickle
import os

from scipy.sparse import data
from scipy.sparse.linalg import spsolve, eigsh

# Unidades Base
m = 1
kg = 1
s = 1
# Otras Unidades
cm = 0.01 * m
N = kg * m / s**2
kN = 1000 * N
kgf = 9.80665 * N
tonf = 1000 * kgf
Pa = N / m**2
MPa = 10**6 * Pa
# Constantes Físicas
g = 9.80665 * m / s**2


def open_data():
    data_file = open('./Data', 'rb')
    data = pickle.load(data_file)
    return data


def init_model():
    try:
        os.remove('Data')
    except:
        pass


def ProblemData(SpaceDim=2, pde='Elasticity'):

    data = {}
    data['SpaceDim'] = SpaceDim
    data['pde'] = pde

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def ElementData(elemDof=2,
                elemNodes=4,
                elem_noInt=4,
                bb=1,
                bh=1,
                thickness=0.5,
                fy=0,
                elemType="Quad4",
                mass_noInt=4,
                massMat='lumped'):

    data = open_data()

    data['elem_dof'] = elemDof
    data['elem_nodes'] = elemNodes
    data['elem_noInt'] = elem_noInt
    data['bb'] = bb
    data['bh'] = bh
    data['thickness'] = thickness
    data['fy'] = fy

    data['elem_type'] = elemType
    data['mass_noInt'] = mass_noInt
    data['mass_Mat'] = massMat

    data['AreaSec'] = bb * bh
    data['I'] = bb * bh**3 / 12.0

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def ModelData(E=1,
              v=0.25,
              G=0,
              GAs=0,
              density=2400 * kg / m**3,
              selfWeight=0.0,
              gravity=[0.0, -1.0, 0.0]):

    data = open_data()
    # Agregarr a Data los datos del modelo
    data['E'] = E
    data['v'] = v
    data['G'] = G
    data['GAs'] = GAs

    data['density'] = density
    data['selfWeight'] = selfWeight
    data['gravity'] = np.array(gravity)

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def GenLinearMesh(L, N):
    '''
    Define los nodos y conexiones de un Mesh especificado para un elemento finito en 1D.
    Defines the nodes and connections of a specified mesh for a Finite Element in 1D.
    Input
        L   : Longitud total de la estructura.
        L   : Total length of the structure
        N   : Cantidad de elementos en las que se dividirá la estructura     
        N   : Number of element into which the structure will be divided
    '''
    # Número de subelementos
    # Number of sub-elements

    L_element = L / N

    print('=' * 7 + 'Mesh' + '=' * 7)
    print("nx = {}, dx = {:.2f}".format(
        N,
        L_element,
    ))

    # Define la cantidad de nodos
    # Defines the number of nodes

    Nodes = np.zeros(N + 1)

    for i in range(N + 1):

        Nodes[i] = i * L_element

    connect = np.zeros((N, 2), dtype=np.uint8)

    connect[:, 0], connect[:, 1] = range(N), range(1, N + 1)

    data = open_data()
    data['NN'] = len(Nodes)
    data['Nodes'] = Nodes
    data['NC'] = len(connect)
    data['Connect'] = connect
    data['Len_beam'] = L

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def GenQuadMesh_2D(L, H, lc):
    '''Crea la malla de un elemento rectangular con códigos programados en Python.

    Create the mesh of a rectagular element with codes programmed in Python

    L  : Longitud o base del elemento  en metros
    H  : Altura del elemento en metros
    lc : Número de elementos en la dirección más corta entre L o H.

    L  : Element length or base in meters
    H  : Element height in meters
    lc : Number of elements in the direction of the smaller dimension between L or H.
    '''
    # Lectura del archivo data
    # Reading the data file

    data = open_data()
    SpaceDim = data['SpaceDim']

    # Define la cantidad de elementos y sus dimensiones en ambas direcciones.
    # Defines the number of elements and element dimensions in both directions.

    if np.min((L, H)) == L:

        nx = lc
        ms_x = L / nx
        ny = round(H / ms_x)
        ms_y = H / ny

    else:

        ny = lc
        ms_y = H / ny
        nx = round(L / ms_y)
        ms_x = L / nx

    # Imprime los resultados
    # Print the results.

    print('=' * 16 + 'Mesh' + '=' * 16)
    print("nx = {}, dx = {:.2f}, ny = {}, dy = {:.2f}".format(
        nx, ms_x, ny, ms_y))

    # Se definen los nodos de la malla
    # Mesh nodes are defined

    noNodes = (nx + 1) * (ny + 1)
    Nodes = np.zeros((noNodes, 2), dtype=np.float64)

    ni = 0

    for i in range(ny + 1):
        for j in range(nx + 1):

            Nodes[ni] = (ms_x * j, ms_y * i)

            ni = ni + 1

    # Se establecen las conexiones entre los nodos de la malla para definir los elementos finitos
    # The connection bewteen the mesh nodes is set up to define the finite elements

    noElem = nx * ny
    connect = np.zeros((noElem, 4), dtype=np.int32)

    k = 0

    for i in range(0, ny):
        for j in range(0, nx):

            connect[k, 0] = j + (i * (nx + 1))
            connect[k, 1] = j + (i * (nx + 1)) + 1
            connect[k, 2] = j + ((i + 1) * (nx + 1)) + 1
            connect[k, 3] = j + ((i + 1) * (nx + 1))

            k = k + 1

    # Se agrega los parámetros calculados al diccionario Data
    # The parameters obtained are added to the Data dictionary

    data['NN'] = noNodes
    data['NC'] = noElem
    data['Nodes'] = Nodes
    data['Connect'] = connect
    data['nx'] = nx
    data['L'] = L
    data['H'] = H

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def GenBrickMesh_3D(L, B, H, lc):
    '''Crea la malla de un elemento rectangular con códigos programados en Python.

    Create the mesh of a rectagular element with codes programmed in Python

    L  : Longitud o base del elemento  en metros
    H  : Altura del elemento en metros
    lc : Número de elementos en la dirección más corta entre L o H.

    L  : Element length or base in meters
    H  : Element height in meters
    lc : Number of elements in the direction of the smaller dimension between L or H.
    '''
    # Lectura del archivo data
    # Reading the data file

    data = open_data()
    SpaceDim = data['SpaceDim']

    # Define la cantidad de elementos y sus dimensiones en ambas direcciones.
    # Defines the number of elements and element dimensions in both directions.

    if np.min((L, B, H)) == L:

        nx = lc
        ms_x = L / nx
        ny = round(B / ms_x)
        ms_y = B / ny
        nz = round(H / ms_x)
        ms_z = H / nz

    elif np.min((L, B, H)) == B:

        ny = lc
        ms_y = B / ny
        nx = round(L / ms_y)
        ms_x = L / nx
        nz = round(H / ms_y)
        ms_z = H / nz

    else:

        nz = lc
        ms_z = H / nz
        ny = round(B / ms_z)
        ms_y = B / ny
        nx = round(L / ms_z)
        ms_x = L / nx

    # Imprime los resultados
    # Print the results.

    print('=' * 16 + 'Mesh' + '=' * 16)
    print("nx = {}, dx = {:.2f}, ny = {}, dy = {:.2f}, nz = {}, dz = {:.2f}".
          format(nx, ms_x, ny, ms_y, nz, ms_z))

    # Se definen los nodos de la malla
    # Mesh nodes are defined

    noNodes = (nx + 1) * (ny + 1) * (nz + 1)
    Nodes = np.zeros((noNodes, 3), dtype=np.float64)

    ni = 0
    for i in range(nz + 1):
        for j in range(ny + 1):
            for k in range(nx + 1):

                Nodes[ni] = (ms_x * k, ms_y * j, ms_z * i)

                ni = ni + 1

    # Se establecen las conexiones entre los nodos de la malla para definir los elementos finitos
    # The connection bewteen the mesh nodes is set up to define the finite elements

    noElem = nx * ny * nz
    connect = np.zeros((noElem, 8), dtype=np.int32)

    cont = 0
    for k in range(0, nz):
        for i in range(0, ny):
            for j in range(0, nx):

                connect[cont,
                        0] = j + (i * (nx + 1)) + (k * (nx + 1) * (ny + 1))
                connect[cont,
                        1] = j + (i * (nx + 1)) + (k * (nx + 1) * (ny + 1)) + 1
                connect[cont, 2] = j + ((i + 1) * (nx + 1)) + ((k) * (nx + 1) *
                                                               (ny + 1)) + 1
                connect[cont, 3] = j + ((i + 1) * (nx + 1)) + ((k) * (nx + 1) *
                                                               (ny + 1))
                connect[cont, 4] = j + (i * (nx + 1)) + ((k + 1) * (nx + 1) *
                                                         (ny + 1))
                connect[cont, 5] = j + (i * (nx + 1)) + ((k + 1) * (nx + 1) *
                                                         (ny + 1)) + 1
                connect[cont,
                        6] = j + ((i + 1) * (nx + 1)) + ((k + 1) * (nx + 1) *
                                                         (ny + 1)) + 1
                connect[cont,
                        7] = j + ((i + 1) * (nx + 1)) + ((k + 1) * (nx + 1) *
                                                         (ny + 1))

                cont += 1

    # Se agrega los parámetros calculados al diccionario Data
    # The parameters obtained are added to the Data dictionary

    data['NN'] = noNodes
    data['NC'] = noElem
    data['nz'] = nz
    data['Nodes'] = Nodes
    data['Connect'] = connect

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def LinearMesh(L, N, x0=0):
    '''Función que retorna los nodos y conexiones de un Mesh especificado
       para un elemento finito (EF) en 1D.
    '''
    L_element = L / N
    nodos = np.zeros(N + 1)
    for i in range(N + 1):
        nodos[i] = x0 + i * L_element
    conex = np.zeros((N, 2), dtype=np.uint8)
    conex[:, 0], conex[:, 1] = range(N), range(1, N + 1)
    return nodos, conex


def DC_node(xy=(1, 2), dof=(1, 2), lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Dirichlet de un nodo
    Return an array with information to define Dirichlet condition of a node

    xy      : Posición del nodo 
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Node position
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''
    x = xy[0]
    y = xy[1]

    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']

    # Crea el diccionario dic_vector
    # dic_vector dictionary is created

    dic_vector = {}

    for i in dof:

        aux_1 = np.where(abs(Nodes[:, 0] - x) < lim)
        aux_2 = np.where(abs(Nodes[:, 1] - y) < lim)
        aux = np.intersect1d(aux_1, aux_2)

        vector = np.hstack((aux[:, np.newaxis], [[1, i, 0]] * len(aux)))
        dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2)
    # Array elements area grouped vertically (dof=1, dof =2)

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_data' in data.keys():

        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)

    else:

        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def DC_nodes(dir='x', dist=0, dof=[1, 2], lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Dirichlet de un conjunto de nodos en la dirección definida
    Return an array with information to define Dirichlet condition from a set of nodes in a direction defined

    dir     : Dirección en la que se restringirán los nodos
    dist    : Posición del eje de restricción (en metros)
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Direction in which nodes will be restricted
    dist    : Position of the restriction axis (in meters)
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''
    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']

    # Crea el diccionario dic_vector
    # Creating dic_vector dictionary

    dic_vector = {}

    if dir == 'x':

        for i in dof:

            aux = np.where(abs(Nodes[:, 1] - dist) < lim)
            vector = np.hstack((aux[0][:,
                                       np.newaxis], [[1, i, 0]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'y':

        for i in dof:

            aux = np.where(abs(Nodes[:, 0] - dist) < lim)
            vector = np.hstack((aux[0][:,
                                       np.newaxis], [[1, i, 0]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'yz':

        for i in dof:
            aux = np.where(abs(Nodes[:, 0] - dist) < lim)
            vector = np.hstack((aux[0][:,
                                       np.newaxis], [[1, i, 0]] * len(aux[0])))
            dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2)
    # Array elements are grouped vertically (dof=1, dof =2)

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_data' in data.keys():

        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)

    else:

        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def NC_node(xyz=(1, 1), force=0, dof=(1), lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Neumann de un nodo
    Return a array with information to define Neumann condition of a node

    xy      : Posición del nodo 
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Node position
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']
    SpaceDim = data['SpaceDim']

    # Crea el diccionario dic_vector
    # dic_vector dictionary is created

    dic_vector = {}

    if SpaceDim == 2:
        x = xyz[0]
        y = xyz[1]

        for i in dof:

            aux_1 = np.where(abs(Nodes[:, 0] - x) < lim)
            aux_2 = np.where(abs(Nodes[:, 1] - y) < lim)

            aux = np.intersect1d(aux_1, aux_2)

            vector = np.hstack((aux[:,
                                    np.newaxis], [[0, i, force]] * len(aux)))

            dic_vector.update({i: vector})

    elif SpaceDim == 3:

        from functools import reduce

        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        for i in dof:

            aux_1 = np.where(abs(Nodes[:, 0] - x) < lim)
            aux_2 = np.where(abs(Nodes[:, 1] - y) < lim)
            aux_3 = np.where(abs(Nodes[:, 2] - z) < lim)

            aux = reduce(np.intersect1d, (aux_1, aux_2, aux_3))

            vector = np.hstack((aux[:,
                                    np.newaxis], [[0, i, force]] * len(aux)))

            dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2, dof =3)
    # Array elements area grouped vertically (dof=1, dof =2, dof =3)

    vector = np.vstack(tuple(dic_vector.values()))
    if 'BC_data' in data.keys():
        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)
    else:
        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def NC_nodes(dir='x', dist=0, force=1000, dof=(1, 2), lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Neumann de un conjunto de nodos en la dirección definida
    Return a array with information to define Neumann condition of a set of nodes in a defined direction

    dir     : Dirección en la que se restringirán los nodos
    dist    : Posición del eje de restricción (en metros)
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Direction in which nodes will be restricted
    dist    : Position of the restriction axis (in meters)
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']

    # Crea el diccionario dic_vector
    # dic_vector dictionary is created

    dic_vector = {}

    if dir == 'x':

        for i in dof:
            aux = np.where(abs(Nodes[:, 1] - dist) < lim)
            vector = np.hstack(
                (aux[0][:, np.newaxis], [[0, i, force]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'y':

        for i in dof:
            aux = np.where(abs(Nodes[:, 0] - dist) < lim)
            vector = np.hstack(
                (aux[0][:, np.newaxis], [[0, i, force]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'yz':

        for i in dof:
            aux = np.where(abs(Nodes[:, 0] - dist) < lim)
            vector = np.hstack(
                (aux[0][:, np.newaxis], [[0, i, force]] * len(aux[0])))
            dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2)
    # Array elements area grouped vertically (dof=1, dof =2)

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_data' in data.keys():
        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)
    else:
        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def DC_node_beam(x=(0), dof=(1, 2), lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Dirichlet de un nodo
    Return an array with information to define Dirichlet condition of a node

    xy      : Posición del nodo 
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Node position
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']

    # Crea el diccionario dic_vector
    # dic_vector dictionary is created

    dic_vector = {}

    for i in dof:

        aux = np.where(abs(Nodes - x) < lim)

        vector = np.hstack((aux, [[1, i, 0]] * len(aux[0])))
        dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2)
    # Array elements area grouped vertically (dof=1, dof =2)

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_data' in data.keys():

        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)

    else:

        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def NC_node_beam(x=(1, 1), force=0, dof=(1), lim=0.001):
    '''
    Retorna un arreglo con la información para definir las condiciones de  Neumann de un nodo
    Return a array with information to define Neumann condition of a node

    xy      : Posición del nodo 
    gdl     : Grados de libertad que se restringirán
    lim     : Tolerancia en metros

    dir     : Node position
    gld     : Degrees of freedom to restrict
    lim     : tolerance in meters

    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()
    Nodes = data['Nodes']

    # Crea el diccionario dic_vector
    # dic_vector dictionary is created

    dic_vector = {}

    for i in dof:

        aux = np.where(abs(Nodes - x) < lim)

        vector = np.hstack((aux, [[0, i, force]] * len(aux[0])))

        dic_vector.update({i: vector})

    # Se agrupan verticalmente los elementos array (dof=1, dof =2)
    # Array elements area grouped vertically (dof=1, dof =2)

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_data' in data.keys():
        data['BC_data'] = np.unique(np.vstack((data['BC_data'], vector)),
                                    axis=0)
    else:
        data['BC_data'] = np.unique(vector, axis=0)

    # Se agrega los parámetros calculados al diccionario Data
    # The calculate parameters are added to the Data dictionary

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def ShapeFunction(X, gp, elem_type):
    '''
    Define funciones de forma y sus derivadas en coordenadas naturales para un elemento finito de coordenadas X, N(X)=N(xi),dN(X)=dN(xi)*J-1
    Defines Shape functions and their derivatives in natural coordinates for a finite element of coordinates X, N(X) = N(xi), dN(X) = dN(xi)*J-1
    -----------------
    Input:
        X       : Matriz de coordenadas del elemento
        X       : Element coordinate matrix
        gp      : Arreglo que contiene los parametros de la cuadratura de Gauss
        gp      : Array containing Gauss Quadrature parameters
        type    : Tipo de elemento finito
        type    : Finite element type
    Output:
        N       : Matriz de funciones de Forma
        N       : Matrix of Shape Functions
        dN      : Matriz de la derivada de las funciones de Forma
        dN      : Matrix of the derivative of the Shape Function
        ddN     : Matriz de la segunda derivada de las funciones de Forma
        ddN     : Matrix of the second derivative of the Shape Function
        j       : Determinante del Jacobiano para realizar la integración usando el mapeo isoparamétrico
        j       : Jacobian determinant to integrate using isoparametric mapping
        '''
    if elem_type == 'Bar1':

        N, dN, J = np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 1))

        ξ = gp[1]
        N[0, 0], N[0, 1] = -ξ / 2 + 0.5, ξ / 2 + 0.5
        dN[0, 0], dN[0, 1] = -1 / 2, 1 / 2

    elif elem_type == 'BarB':
        N, dN, ddN = np.zeros((1, 4)), np.zeros((1, 4)), np.zeros((1, 4))

        ξ = gp[1]
        le = X[1] - X[0]
        N[0, 0] = le * (1 - ξ)**2 * (1 + ξ) / 8
        N[0, 1] = (1 - ξ)**2 * (2 + ξ) / 4
        N[0, 2] = le * (1 + ξ)**2 * (-1 + ξ) / 8
        N[0, 3] = (1 + ξ)**2 * (2 - ξ) / 4
        dN[0, 0] = -(1 - ξ) * (3 * ξ + 1) / 4
        dN[0, 1] = -3 * (1 - ξ) * (1 + ξ) / (2 * le)
        dN[0, 2] = (1 + ξ) * (3 * ξ - 1) / 4
        dN[0, 3] = 3 * (1 - ξ) * (1 + ξ) / (2 * le)
        ddN[0, 0] = (3 * ξ - 1) / le
        ddN[0, 1] = 6 * ξ / (le**2)
        ddN[0, 2] = (3 * ξ + 1) / le
        ddN[0, 3] = -6 * ξ / (le**2)
        j = le / 2

    elif elem_type == 'Quad4':

        N, dN, J = np.zeros((1, 4)), np.zeros((2, 4)), np.zeros((2, 2))
        a = np.array(
            [-1.0, 1.0, 1.0, -1.0]
        )  # coordenadas x de los nodos   ########################################## Verificar nombre x o ξ
        b = np.array([-1.0, -1.0, 1.0, 1.0])  # coordenadas y de los nodos

        ξ = gp[1]
        η = gp[2]

        N = 0.25 * (1.0 + a[:] * ξ) * (1.0 + b[:] * η)  # 0.25(1+ξiξ)(1+ηiη)

        dN[0] = 0.25 * a[:] * (1 + b[:] * η)  # dN,ξ = 0.25ξi(1+ηiη)
        dN[1] = 0.25 * b[:] * (1 + a[:] * ξ)  # dN,η = 0.25ηi(1+ξiξ)

    elif elem_type == 'Brick8':

        N, dN, J = np.zeros((3, 8)), np.zeros((3, 8)), np.zeros((3, 3))
        a = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
        b = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        c = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        ξ = gp[1]
        η = gp[2]
        ζ = gp[3]

        N = 0.125 * (1.0 + a[:] * ξ) * (1.0 + b[:] * η) * (1.0 + c[:] * ζ)

        dN[0] = 0.125 * a[:] * (1 + b[:] * η) * (1.0 + c[:] * ζ)
        dN[1] = 0.125 * b[:] * (1 + a[:] * ξ) * (1.0 + c[:] * ζ)
        dN[2] = 0.125 * c[:] * (1 + a[:] * ξ) * (1.0 + b[:] * η)

    else:

        print("Debes programar para el tipo %s aún" % elem_type)

    # Calcula la matriz Jacobiana y su determinante
    # Calculates Jacobian matriz and its determinant

    try:
        j
    except NameError:
        J = dN @ X
        if len(J) > 1:
            j = det(J)
            dN = inv(J) @ dN
        else:
            j = J[0]
            dN = dN / j
    if j < 0:
        print("Cuidado: El jacobiano es negativo!")
        # print(X,'\n',dN,'\n',J,'\n',j)
    if elem_type != 'BarB':
        ddN = 0.0  # Retorna una matriz de ceros cuando no es Bernoulli
    #
    return N, dN, ddN, j


def Bernoulli(A, X, N, dN, ddN, dX, matrix_type):
    '''
    Retorna la matriz de un EF evaluado en un Punto de Gauss
    Return the matrix of the EF evaluated at a Gauss Point

    Input:

        X               : Arreglo que contiene las coordenadas del EF.
        X               : Array containing the coordinates of the EF.
        N               : Arreglo que contiene las funciones de forma.
        N               : Array containing the Shape Functions
        dN              : Arreglo que contiene las derivadas parciales de las funciones de forma.
        dN              : Array containing the partial derivate of the Shape functions
        ddN             : Arreglo que contiene las segundas derivadas parciales de las funciones de forma.
        ddN             : Array containing the second partial derivates of the Shape function 
        dX              :   
        matriz_type     : Texto que indica que arreglo se quiere obtener. Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada
        matriz_type     : Texto que indica que arreglo se quiere obtener. Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada

        
    Output:

        A       : Matriz de elemento finito
        A       : Finite element matrix
    '''
    data = open_data()

    n = data['elem_nodes']
    m = data['elem_dof']
    E = data['E']
    I = data['I']
    bb = data['bb']
    bh = data['bh']
    Area = bb * bh
    EI = E * I
    ρ = data['density']
    fy = data['fy']

    if matrix_type == 'MatrixK':

        A = EI * ddN.T @ ddN
        A = A * dX

    elif matrix_type == 'ConsistentMass':

        A = ρ * N.T @ N * dX * Area
    #
    elif matrix_type == 'ConcentratedMass':

        B = ρ * N.T @ N * dX * Area
        one = np.zeros(m * n) + 1.0
        B = B @ one
        A = np.zeros((m * n, m * n))

        # Concentrando Masas (Mejorar proceso si |A| <= 0)

        for i in range(m * n):
            A[i, i] = B[i]
    #
    elif matrix_type == 'VectorF':

        # Formando matriz N para v

        N_v = np.zeros((1, m * n))

        N_v = N

        A = np.zeros((1, m * n))

        f = fy

        A = N_v * f * dX
        ##
    else:

        print("Debes programar para el tipo %s aún" % matrix_type)

    return A


def Timoshenko(A, X, N, dN, ddN, dX, matrix_type):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    '''
    data = open_data()

    n = data['elem_nodes']
    m = data['elem_dof']
    E = data['E']
    I = data['I']
    EI = E * I
    GAs = data['GAs']
    GAs = GAs
    bb = data['bb']
    bh = data['bh']
    fy = data['fy']

    Area = bb * bh
    fy = data['fy']
    ρ = data['density']

    SpaceDim = data['SpaceDim']

    ##
    if matrix_type == 'MatrixK':

        if (SpaceDim == 1):
            # Formando N para theta
            N_theta = np.zeros((1, m * n))
            N_theta[0, 0::m] = N[0]
            # Formando N para v
            N_v = np.zeros((1, m * n))
            N_v[0, 1::m] = N[0]
            # Formando B para theta
            B_theta = np.zeros((1, m * n))
            B_theta[0, 0::m] = dN[0]
            # Formando B para for v
            B_v = np.zeros((1, m * n))
            B_v[0, 1::m] = dN[0]
            #
        elif (SpaceDim == 2):
            print('Solo es válido para 1D')
        elif (SpaceDim == 3):
            print('Solo es válido para 1D')
            #
        # Calculando Matriz
        A = B_theta.T * EI @ B_theta + N_theta.T * GAs @ N_theta - N_theta.T * GAs @ B_v
        A = A - B_v.T * GAs @ N_theta + B_v.T * GAs @ B_v
        A = A * dX

    elif matrix_type == 'ConsistentMass':

        Nv = np.zeros((1, m * n))
        Nv[0, 0], Nv[0, 1], Nv[0, 2], Nv[0, 3] = 0, N[0, 0], 0, N[0, 1]
        A = ρ * Nv.T @ Nv * dX * Area
        #
        #  Artificio para no obtener una Matriz singular
        A[0, 0] = A[0, 0] + 0.01 * A[1, 1]
        A[2, 2] = A[2, 2] + 0.01 * A[3, 3]

    elif matrix_type == 'ConcentratedMass':

        nulo = np.array([0])
        Nv[0, 0], Nv[0, 1], Nv[0, 2], Nv[0, 3] = 0, N[0, 0], 0, N[0, 1]
        B = ρ * Nv.T @ Nv * dX * Area
        # Artificio para no obtener una Matriz singular
        B[0, 0] = B[0, 0] + 0.01 * B[1, 1]
        B[2, 2] = B[2, 2] + 0.01 * B[3, 3]
        #
        one = np.zeros(m * n) + 1.0
        B = B @ one
        A = np.zeros((m * n, m * n))
        # Concentrando Masas
        for i in range(m * n):
            A[i, i] = B[i]

    elif (matrix_type == 'VectorF'):

        # Formando matriz N para theta
        N_theta = np.zeros((1, m * n))
        N_theta[0, 0::m] = N[0]
        # Formando matriz N para v
        N_v = np.zeros((1, m * n))
        N_v[0, 1::m] = N[0]
        #
        f = fy
        A = N_v * f * dX

    else:
        print("Debes programar para el tipo %s aún" % matrix_type)
    return A


def GaussQuadrature(point, dim):
    '''
    Define los puntos de integración según la cuadratura de Gauss
    Defines the Integration Points according to the Gauss Quadrature

    ---------------
    Input:
            point   : El número de puntos de integración de Gauss
            point   : Number of Gauss integration points
            dim     : Dimensión del elemento finito
            dim     : Finite element dimension
    Output:
            gp      : Arreglo cuya primera fila son los pesos y las demás las posiciones respectivas
            gp      : Array whose first row are the weights and the others the respective positions                  
    '''
    if dim == 1:

        if point == 1:  # Number of integration points = 1 Dimension = 1D

            gp = np.zeros((2, 1))
            gp[0], gp[1] = 2.0, 0.0

        elif point == 2:  # Number of integration points = 2 Dimension = 1D

            a = 1.0 / 3**0.5
            gp = np.zeros((2, 2))
            gp[0, :] = 1.0
            gp[1, 0], gp[1, 1] = -a, a

        elif point == 4:  # Number of integration points = 4 Dimension = 1D

            gp = np.zeros((2, 4))
            a, b = 0.33998104358484, 0.8611363115941
            wa, wb = 0.65214515486256, 0.34785484513744
            gp[0, 0], gp[0, 1], gp[0, 2], gp[0, 3] = wb, wa, wa, wb
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1, 3] = -b, -a, a, b

        else:

            print("Debes programar para %s puntos aún." % point)

    elif dim == 2:

        if point == 4:  # Number of integration points = 4 Quad4

            a = 1.0 / 3**0.5

            gp = np.zeros((3, 4))

            gp[0, :] = 1.0  # Weight
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1,
                                             3] = -a, a, a, -a  # Position in ξ
            gp[2, 0], gp[2, 1], gp[2, 2], gp[2,
                                             3] = -a, -a, a, a  # Position in η
    elif dim == 3:

        if point == 4:
            a = 1.0 / 3**0.5

            gp = np.zeros((4, 8))
            gp[0, :] = 1.0
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1, 3], gp[1, 4], gp[1, 5], gp[
                1, 6], gp[1, 7] = -a, a, a, -a, -a, a, a, -a
            gp[2, 0], gp[2, 1], gp[2, 2], gp[2, 3], gp[2, 4], gp[2, 5], gp[
                2, 6], gp[2, 7] = -a, -a, a, a, -a, -a, a, a
            gp[3, 0], gp[3, 1], gp[3, 2], gp[3, 3], gp[3, 4], gp[3, 5], gp[
                3, 6], gp[3, 7] = -a, -a, -a, -a, a, a, a, a

        else:

            print("Debes programar para %s puntos aún" % point)

    else:

        print("Debes programar para %sD aún" % dim)

    return gp.T


def Elasticity(A, X, N, dN, dNN, dX, matrix_type):
    '''
    Retorna la matriz de un EF evaluado en un Punto de Gauss
    Return the matrix of the EF evaluated at a Gauss Point

    Input:

        X               : Arreglo que contiene las coordenadas del EF.
        X               : Array containing the coordinates of the EF.
        N               : Arreglo que contiene las funciones de forma.
        N               : Array containing the Shape Functions
        dN              : Arreglo que contiene las derivadas parciales de las funciones de forma.
        dN              : Array containing the partial derivate of the Shape functions
        ddN             : Arreglo que contiene las segundas derivadas parciales de las funciones de forma.
        ddN             : Array containing the second partial derivates of the Shape function 
        dX              :   
        matriz_type     : Texto que indica que arreglo se quiere obtener. Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada
        matriz_type     : Texto que indica que arreglo se quiere obtener. Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada

        
    Output:

        A       : Matriz de elemento finito
        A       : Finite element matrix
    '''

    data = open_data()

    n = data['elem_nodes']
    m = data['elem_dof']
    t = data['thickness']

    SpaceDim = data['SpaceDim']
    density = data['density']
    selfweight = data['selfWeight']
    gravity = data['gravity']

    if matrix_type == 'MatrixK':

        E, v = data['E'], data['v']
        E, v = E, v  # Estado plano de esfuerzos Flat state of stress
        # E,v = E/(1.0-v*2),v/(1-v)# Estado plano de deformaciones Flat state of deformations

        if SpaceDim == 2:

            # Formando Matriz D
            # Forming Matriz D

            D = np.zeros((3, 3))

            D[0, 0], D[1, 1], D[0, 1], D[1, 0] = 1.0, 1.0, v, v

            D[2, 2] = 0.5 * (1.0 - v)

            D = E * D / (1 - v**2)

            # Formando Matriz B
            # Forming Matrix B

            B = np.zeros((3, m * n))

            for i in range(m):

                B[i, i::m] = dN[i]

            B[2, 0::m] = dN[1]
            B[2, 1::m] = dN[0]

        elif SpaceDim == 3:

            # Formando Matriz D
            # Forming Matriz D

            D = np.zeros((6, 6))
            λ = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
            μ = E / (2.0 * (1.0 + v))

            D[0, 0] = D[1, 1] = D[2, 2] = μ + 2 * λ
            D[3, 3] = D[4, 4] = D[5, 5] = μ
            D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = λ

            # Formando Matriz B
            # Forming Matrix B

            B = np.zeros((6, m * n))

            for i in range(m):

                B[i, i::m] = dN[i]

            B[2, 0::m] = dN[1]
            B[2, 1::m] = dN[0]
            B[3, 1::m] = dN[2]
            B[3, 2::m] = dN[1]
            B[4, 0::m] = dN[2]
            B[4, 2::m] = dN[0]

        else:
            print("Debes programar para %sD aún" % SpaceDim)

        A = B.T @ D @ B * dX * t

    elif matrix_type == 'ConsistentMass':

        Nmat = np.zeros((m, m * n))

        rho = density

        for i in range(m):

            Nmat[i, i::m] = N[:]

        A = rho * Nmat.T @ Nmat * dX * t

    elif matrix_type == 'ConcentratedMass':

        Nmat = np.zeros((m, m * n))

        rho = density
        for i in range(m):

            Nmat[i, i::m] = N[:]

        B = rho * Nmat.T @ Nmat * dX * t

        one = np.zeros(m * n) + 1.0

        B = B @ one

        A = np.zeros((m * n, m * n))

        for i in range(m * n):

            A[i, i] = B[i]

    elif matrix_type == 'VectorF':

        Nmat = np.zeros((m, m * n))

        for i in range(m):

            Nmat[i, i::m] = N[:].T

            f = selfweight * gravity[0:m]

            A = Nmat.T @ f * dX * t

    else:

        print("Debes programar para el tipo %s aún" % matrix_type)

    return A


def DofMap(DofNode, connect, NodesElement):
    '''
    Función que mapea los grados de libertad correspondientes a un EF

    Input:

            DofNode     : Cantidad de grados de libertad por nodo
            DofNode     : Amount of degree of freedom per node
            connect     : Nodos del Elemento Finito
            connect     : Finite element nodes
            NodesElement: Cantidad de nodos del Elemento Finito
            NodesElement: Amount of nodes of the Finite Element
            
    Output:

            dof         : Lista que contiene los grados de libertad del EF en el sistema global.
            dof:        : List that containing the degrees of freedom of the EF in the global system.

    '''
    dof = []

    for i in range(NodesElement):

        for j in range(DofNode):

            dof.append(DofNode * connect[i] + j)

    return dof


def AssembleMatrix(MatrixType):
    '''
    Realiza el ensamble de la matriz K, de Ku = F
    Performs the assembly of the matrix k, of Ku = F
    

        Input:
                MatrixType  : Texto que indica que arreglo se quiere obtener
                MatrixType  : Text indicating the array to obtain
        Output:
                A           : Arreglo obtenido del ensamble de las matrices de los elementos finitos.
                A           : Array obtained from assembly of finite elements matrix.
    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()

    NN = data['NN']
    elem_dof = data['elem_dof']
    elem_type = data['elem_type']
    elem_noInt = data['elem_noInt']
    elem_nodes = data['elem_nodes']
    Nodes = data['Nodes']
    SpaceDim = data['SpaceDim']
    Connect = data['Connect']
    PDE = data['pde']

    if MatrixType == 'MatrixK':
        elem_noInt = data['elem_noInt']
    else:
        elem_noInt = data['mass_noInt']

    # Calcula el número de grados de libertad totales y el número de grados de libertad por cada elemento
    # Calculates the total number of degrees of freedom and the number of degree of freedom per element

    N = NN * elem_dof
    n = elem_nodes * elem_dof

    # Definine la Matriz "SPARSE" para la Matriz Global con el número total de grados de libertad del modelo
    # Defines the "SPARSE" Matrix for the Global Matrix with the total number of degree of freedom of the model

    A = sparse.lil_matrix((N, N), dtype=np.float64)

    # Define la Matriz para los elementos
    # Defines the matrix for the elements

    A_e = np.zeros(n, dtype=np.float64)
    A_int = np.zeros(n, dtype=np.float64)

    # Obtiene los pesos y posiciones de la Cuadratura de Gauss
    # Gets the weight and position of the Gauss Quadrature

    gp = GaussQuadrature(elem_noInt, SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    # Loop to assembly the matriz of each element

    for connect_element in Connect:

        # Asigna las coordenadas de nodos y su connectividad
        # Node coordinates and connectivity are assigned

        x_element = Nodes[connect_element]

        # Bucle para realizar la integración según Cuadratura de Gauss
        # Loop to integrate according to Gauss Quadrature

        for gauss_point in gp:

            # Calcula las Funciones de Forma
            # Calculates the Shape Funtions
            [N, dN, ddN, j] = ShapeFunction(x_element, gauss_point, elem_type)

            # Calculando la Matriz de cada Elemento
            # Computing the matrix of each element

            dX = gauss_point[0] * j

            # Evalua el PDE (Elasticidad, Timoshenko, Bernoulli, etc.)
            # Evaluate the PDE (Elasticity, Timoshenko, Bernoulli, etc.)
            A_int = eval(PDE + '(A_int, x_element, N, dN,ddN, dX, MatrixType)')
            A_e = A_e + A_int

        # Mapea los grados de libertad
        # Degrees of freedom mapping

        dof = DofMap(elem_dof, connect_element, elem_nodes)

        # Ensamblando
        # Assembling

        cont = 0

        for k in dof:

            A[k, dof] = A[k, dof] + A_e[cont]

            cont = cont + 1

        # Restablece la Matriz del Elemento
        # Resets the matrix of the element

        A_e[:] = 0.0

    if MatrixType == "MatrixK":

        data['K'] = A

        print("Matriz K calculada")

        data_file = open('./Data', 'wb')
        pickle.dump(data, data_file)
        data_file.close()

    elif MatrixType == "ConcentratedMass":

        data['M'] = A

        print("Matriz M calculada")

        data_file = open('./Data', 'wb')
        pickle.dump(data, data_file)
        data_file.close()

    elif MatrixType == "ConsistentMass":

        data['M'] = A

        print("Matriz M calculada")

        data_file = open('./Data', 'wb')
        pickle.dump(data, data_file)
        data_file.close()


def AssembleVector():
    '''
    Realiza el ensamble del vector F, de Ku=F
    Performs the assembly of the F vector from Ku = F

    Output:
        f   : Vector obtenido del ensamble de los vectores f_e de los elementos finitos.
        f   : Vector obtained from the assembly of the vectors f_e of the finite elements.
    '''

    # Lee el archivo Data
    # Reading Data file

    data = open_data()

    NN = data['NN']
    elem_dof = data['elem_dof']
    elem_type = data['elem_type']
    elem_noInt = data['elem_noInt']
    Nodes = data['Nodes']
    SpaceDim = data['SpaceDim']
    Connect = data['Connect']
    elem_nodes = data['elem_nodes']
    PDE = data['pde']

    # Calcula el número de grados de libertad totales y el número de grados de libertad por cada elemento
    # Calculates the total number of degrees of freedom and the number of degree of freedom per element

    N = NN * elem_dof
    n = elem_nodes * elem_dof

    # Define el vector f global
    # Defines the f global vector

    f = np.zeros(N, np.float64)

    # Define Vector f_e de los elementos
    # Defines Vector f_e of elements

    f_e = np.zeros(n, np.float64)
    f_int = np.zeros(n, np.float64)

    # Obtiene los pesos y posiciones de la Cuadratura de Gauss
    # Gets the weight and position of the Gauss Quadrature

    gp = GaussQuadrature(elem_noInt, SpaceDim)

    # Bucle para ensamblar la matriz de cada elemento
    # Loop to assembly the matriz of each element

    for connect_element in Connect:

        # Asigna las coordenadas de nodos y su connectividad
        # Node coordinates and connectivity are assigned

        x_element = Nodes[connect_element]

        # Bucle para realizar la integración según Cuadratura de Gauss
        # Loop to integrate according to Gauss Quadrature

        for gauss_point in gp:

            # Calcula las Funciones de Forma
            # Calculates the Shape Funtions

            [N, dN, ddN, j] = ShapeFunction(x_element, gauss_point, elem_type)

            # Calculando la Matriz de cada Elemento
            # Computing the matrix of each element

            dX = gauss_point[0] * j

            # Evalua el PDE (Elasticidad, Timoshenko, Bernoulli, etc.)
            # Evaluate the PDE (Elasticity, Timoshenko, Bernoulli, etc.)

            f_int = eval(PDE + '(f_int, x_element, N, dN, ddN, dX, "VectorF")')
            f_e = f_e + f_int

        # Mapea los grados de libertad
        # Degrees of freedom mapping

        dof = DofMap(elem_dof, connect_element, elem_nodes)

        # Ensamblando
        # Assembling

        f[dof] = f[dof] + f_e

        f_e = 0.0
    data['f'] = f

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def ApplyBC():
    '''
    Asigna las condiciones de borde especificadas
    Assigns the specified edge conditions

    '''

    data = open_data()

    elem_dof = data['elem_dof']
    BC_data = data['BC_data']
    A = data['K']
    f = data['f']
    dof_to_reduce = []

    for bc in BC_data:

        if int(bc[1]) == 0:  # Neumann

            dof = int(elem_dof * bc[0] + bc[2]) - 1

            print("CB Neumann, DOF:", dof)

            f[dof] = f[dof] + bc[3]

        elif int(bc[1]) == 1:  # Dirichlet

            dof = int(elem_dof * bc[0] + bc[2]) - 1

            dof_to_reduce.append(dof)

            print("CB Dirichlet, DOF:", dof)

            A[dof, :] = 0.0
            A[dof, dof] = 1.0
            f[dof] = bc[3]

        else:

            print('Condición de Borde Desconocida')

    data['K'] = A
    data['f'] = f
    data['dof_to_reduce'] = dof_to_reduce

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def Analysis():

    data = open_data()
    K = data['K']
    f = data['f']

    u = spsolve(K.tocsr(), f)

    data['u'] = u
    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def deformed(u, FS=1):
    ''' Función que agrega una deformación a la pocisión de Nodos
        Input:
            u       : Arreglo que contiene las deformaciones calculadas con el MEF 
            u       : Array containing the deformations calculated with FEM.
            FS      : Factor de Escala
            FS      : Scale factor
        Output:
            X_def   : Arreglo que contiene la deformada del sistema
            X_def   : Array containing the systema deformed
            X_incr  : Arreglo que contiene los desplazamientos de los nodos
            X_incr  : Array containing the displacements of the nodes
        '''
    data = open_data()

    Nodes = data['Nodes']
    SpaceDim = data['SpaceDim']

    NN = len(Nodes)

    X_incr = np.zeros(Nodes.shape)
    X_def = np.zeros(Nodes.shape)

    if SpaceDim == 2:
        for i in range(NN):
            X_incr[i] = FS * np.array([u[2 * i], u[2 * i + 1]])
            X_def[i] = Nodes[i] + FS * np.array([u[2 * i], u[2 * i + 1]])

    elif SpaceDim == 3:
        for i in range(NN):
            X_incr[i] = FS * np.array([u[3 * i], u[3 * i + 1], u[3 * i + 2]])
            X_def[i] = Nodes[i] + X_incr[i]

    return X_def, X_incr


def deformed_din(Nodes, u, FS=1):
    ''' Función que agrega una deformación a la pocisión de Nodos
        Input:
            u       : Arreglo que contiene las deformaciones calculadas con el MEF 
            u       : Array containing the deformations calculated with FEM.
            FS      : Factor de Escala
            FS      : Scale factor
        Output:
            X_def   : Arreglo que contiene la deformada del sistema
            X_def   : Array containing the systema deformed
            X_incr  : Arreglo que contiene los desplazamientos de los nodos
            X_incr  : Array containing the displacements of the nodes
        '''
    data = open_data()

    SpaceDim = data['SpaceDim']

    NN = len(Nodes)

    X_incr = np.zeros(Nodes.shape)
    X_def = np.zeros(Nodes.shape)

    if SpaceDim == 2:
        for i in range(NN):
            X_incr[i] = FS * np.array([u[2 * i], u[2 * i + 1]])
            X_def[i] = Nodes[i] + FS * np.array([u[2 * i], u[2 * i + 1]])

    elif SpaceDim == 3:
        for i in range(NN):
            X_incr[i] = FS * np.array([u[3 * i], u[3 * i + 1], u[3 * i + 2]])
            X_def[i] = Nodes[i] + X_incr[i]

    return X_def, X_incr


def tridiag(a=2.1, n=5):
    ''' Función que retorna una matriz de rigidez uniforme de a para n gdl
    '''
    aa = [-a for i in range(n - 1)]
    bb = [2 * a for i in range(n)]
    bb[-1] = a
    cc = [-a for i in range(n - 1)]
    return np.diag(aa, -1) + np.diag(bb, 0) + np.diag(cc, 1)

    # def K_reductor(K):
    '''Función que elimina grados de libertad que no se desea analizar
    '''
    data = open_data()
    dof = data['dof_to_reduce']
    k = 0
    for i in dof:
        K = np.delete(np.delete(K, i - k, 0), i - k, 1)
        k = k + 1
    return K


def Reduc_Matrix():
    '''
    Elimina grados de libertad que no se desea analizar

    '''
    data = open_data()

    dof = data['dof_to_reduce']
    K1 = np.array(data['K'].todense())
    M1 = np.array(data['M'].todense())

    # Elimina grados de libertad (Stiffness Matrix)
    # Remove degree of freedom

    Kr = np.delete(np.delete(K1, dof, axis=0), dof, axis=1)

    # Elimina grados de libertad (Mass Matrix)
    # Remove degree of freedom

    Mr = np.delete(np.delete(M1, dof, axis=0), dof, axis=1)

    data['Kr'] = Kr
    data['Mr'] = Mr

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def V_insertor(V):
    '''
    Función que agrega valores nulos aun vector en posiciones especificadas

    '''
    data = open_data()
    dof = data['dof_to_reduce']

    for i in dof:
        V = np.insert(V, i, 0)
    return V


def quads_to_tris(quads):
    # Define la arreglo que contendrá los elementos triangulares
    # Defines the array that will contain the triangular elements

    tris = [[None for j in range(3)] for i in range(2 * len(quads))]

    for i in range(len(quads)):
        j = 2 * i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return tris


def plot_model_mesh():

    data = open_data()

    NC = data['NC']
    Nodes = data['Nodes']
    Connect = data['Connect']

    ##
    import matplotlib.tri as tri
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    elements_quads = Connect
    # FEM data
    nodes_x = Nodes[:, 0]
    nodes_y = Nodes[:, 1]

    # plot the finite element mesh
    for element in elements_quads:
        x = nodes_x[element]
        y = nodes_y[element]
        plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.2)

    plt.axis('equal')
    plt.show()
    plt.savefig('Mesh.png')
    plt.close()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_model_mesh_3D():

    data = open_data()

    NC = data['NC']
    nz = data['nz']
    Nodes = data['Nodes']
    Connect = data['Connect']

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # elements_quads = Connect
    # # FEM data
    nodes_x = np.array(Nodes[:, 0])
    nodes_y = np.array(Nodes[:, 1])
    nodes_z = np.array(Nodes[:, 2])

    for conn in Connect:

        xx_1 = np.concatenate((nodes_x[conn[:4]], [nodes_x[conn[0]]]))
        yy_1 = np.concatenate((nodes_y[conn[:4]], [nodes_y[conn[0]]]))
        zz_1 = np.concatenate((nodes_z[conn[:4]], [nodes_z[conn[0]]]))

        xx_2 = []
        yy_2 = []
        zz_2 = []
        xx_3 = []
        yy_3 = []
        zz_3 = []

        for j in [0, 3, 7, 4, 0]:
            xx_2 = np.concatenate((xx_2, [nodes_x[conn[j]]]))
            yy_2 = np.concatenate((yy_2, [nodes_y[conn[j]]]))
            zz_2 = np.concatenate((zz_2, [nodes_z[conn[j]]]))

        for k in [1, 2, 6, 5, 1]:
            xx_3 = np.concatenate((xx_3, [nodes_x[conn[k]]]))
            yy_3 = np.concatenate((yy_3, [nodes_y[conn[k]]]))
            zz_3 = np.concatenate((zz_3, [nodes_z[conn[k]]]))

        kwargs = {'alpha': 1, 'color': 'blue'}

        ax1.plot3D(xx_1, yy_1, zz_1, **kwargs)
        ax1.plot3D(xx_2, yy_2, zz_2, **kwargs)
        ax1.plot3D(xx_3, yy_3, zz_3, **kwargs)

    for conn in Connect[-int(NC / nz):]:

        xx_1 = np.concatenate((nodes_x[conn[4:]], [nodes_x[conn[4]]]))
        yy_1 = np.concatenate((nodes_y[conn[4:]], [nodes_y[conn[4]]]))
        zz_1 = np.concatenate((nodes_z[conn[4:]], [nodes_z[conn[4]]]))

        ax1.plot3D(xx_1, yy_1, zz_1, **kwargs)

    set_axes_equal(ax1)

    plt.show()

    plt.savefig('Mesh_3D.png')
    plt.close()


def plt_model_deform_1D(FS=1):
    data = open_data()

    from scipy.interpolate import make_interp_spline

    u = data['u']
    xB = data['Nodes']
    yB = u[1::2]

    xB_new = np.linspace(min(xB), max(xB), 100)
    a_BSpline = make_interp_spline(xB, yB)
    yB_new = a_BSpline(xB_new)

    import matplotlib.pyplot as plt
    plt.plot(yB_new, xB_new, 'k-')
    plt.plot(yB, xB, 'ro')
    plt.show()


def plot_model_deform_3D(FS=1000):

    data = open_data()

    NC = data['NC']
    nz = data['nz']
    Connect = data['Connect']
    u = data['u']

    X_Def, X_incr = deformed(u, FS)

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # elements_quads = Connect
    # # FEM data

    nodes_x = np.array(X_Def[:, 0])
    nodes_y = np.array(X_Def[:, 1])
    nodes_z = np.array(X_Def[:, 2])

    for conn in Connect:

        xx_1 = np.concatenate((nodes_x[conn[:4]], [nodes_x[conn[0]]]))
        yy_1 = np.concatenate((nodes_y[conn[:4]], [nodes_y[conn[0]]]))
        zz_1 = np.concatenate((nodes_z[conn[:4]], [nodes_z[conn[0]]]))

        xx_2 = []
        yy_2 = []
        zz_2 = []
        xx_3 = []
        yy_3 = []
        zz_3 = []

        for j in [0, 3, 7, 4, 0]:
            xx_2 = np.concatenate((xx_2, [nodes_x[conn[j]]]))
            yy_2 = np.concatenate((yy_2, [nodes_y[conn[j]]]))
            zz_2 = np.concatenate((zz_2, [nodes_z[conn[j]]]))

        for k in [1, 2, 6, 5, 1]:
            xx_3 = np.concatenate((xx_3, [nodes_x[conn[k]]]))
            yy_3 = np.concatenate((yy_3, [nodes_y[conn[k]]]))
            zz_3 = np.concatenate((zz_3, [nodes_z[conn[k]]]))

        kwargs = {'alpha': 1, 'color': 'blue'}

        ax1.plot3D(xx_1, yy_1, zz_1, **kwargs)
        ax1.plot3D(xx_2, yy_2, zz_2, **kwargs)
        ax1.plot3D(xx_3, yy_3, zz_3, **kwargs)

    for conn in Connect[-int(NC / nz):]:

        xx_1 = np.concatenate((nodes_x[conn[4:]], [nodes_x[conn[4]]]))
        yy_1 = np.concatenate((nodes_y[conn[4:]], [nodes_y[conn[4]]]))
        zz_1 = np.concatenate((nodes_z[conn[4:]], [nodes_z[conn[4]]]))

        ax1.plot3D(xx_1, yy_1, zz_1, **kwargs)

    set_axes_equal(ax1)

    plt.show()

    plt.savefig('Model_Deform_Mesh_3D.png')
    plt.close()


def plot_model_deform(dir='x', FS=1):

    import matplotlib.cm

    data = open_data()

    Connect = data['Connect']
    u = data['u']

    X_Def, X_incr = deformed(u, FS)

    # Posición de los nodos en la estructura deformada
    # Node position in the deformated structure

    nodes_x = X_Def[:, 0]
    nodes_y = X_Def[:, 1]

    # Define en que dirección se mostrarán los resultados
    # Defines in which direcction the results will be displayed

    if dir == 'x':
        nodal_values = X_incr[:, 0]
    else:
        nodal_values = X_incr[:, 1]

    # Define los elementos triangulares a partir de elementos rectangulares
    # Defines Triangle elements from quadrangular elements

    elements_quads = Connect

    elements_all_tris = quads_to_tris(elements_quads)

    # Plotea los elementos finitos
    # plot the finite element mesh

    for element in elements_quads:
        x = nodes_x[element]
        y = nodes_y[element]
        plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.2)

    # create an unstructured triangular grid instance

    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)

    # Plotea los contornos de los elementos finitos
    # plot the contours

    plt.tricontourf(triangulation, nodal_values, cmap="RdBu_r")

    norm = colors.Normalize(vmin=min(nodal_values / FS),
                            vmax=max(nodal_values / FS))

    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                 orientation='horizontal',
                 label='Deformación (m)')
    # Hides axis
    # Ocultar los valores de los ejes
    plt.axis('off')

    plt.axis('equal')

    plt.savefig('Model_Deformada.png')
    plt.show()
    plt.close()


def plot_modes(mode=1, FS=1):

    import matplotlib.cm

    data = open_data()

    Connect = data['Connect']
    vecs = data['vecs'][:, mode - 1]

    X_Def, X_incr = deformed(vecs, FS)

    # Posición de los nodos en la estructura deformada
    # Node position in the deformated structure

    nodes_x = X_Def[:, 0]
    nodes_y = X_Def[:, 1]

    # Define en que dirección se mostrarán los resultados
    # Defines in which direcction the results will be displayed

    if dir == 'x':
        nodal_values = X_incr[:, 0]
    else:
        nodal_values = X_incr[:, 1]

    # Define los elementos triangulares a partir de elementos rectangulares
    # Defines Triangle elements from quadrangular elements

    elements_quads = Connect

    elements_all_tris = quads_to_tris(elements_quads)

    # Plotea los elementos finitos
    # plot the finite element mesh

    for element in elements_quads:
        x = nodes_x[element]
        y = nodes_y[element]
        plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.2)

    # create an unstructured triangular grid instance

    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)

    # Plotea los contornos de los elementos finitos
    # plot the contours

    plt.tricontourf(triangulation, nodal_values, cmap="RdBu_r")

    norm = colors.Normalize(vmin=min(nodal_values / FS),
                            vmax=max(nodal_values / FS))

    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                 orientation='horizontal',
                 label='Deformación (m)')
    # Hides axis
    # Ocultar los valores de los ejes
    plt.axis('off')

    plt.axis('equal')

    plt.savefig('Model_Modes.png')
    plt.show()
    plt.close()


def plt_modes_1D(mode=1, FS=1):

    data = open_data()

    from scipy.interpolate import make_interp_spline

    vecs = data['vecs'][mode - 1]

    xB = data['Nodes']
    yB = np.hstack(([0], vecs[1::2]))

    xB_new = np.linspace(min(xB), max(xB), 100)
    a_BSpline = make_interp_spline(xB, yB)
    yB_new = a_BSpline(xB_new)

    import matplotlib.pyplot as plt
    plt.plot(yB_new, xB_new, 'k-')
    plt.plot(yB, xB, 'ro')
    plt.show()


def Modal_Analysis():

    from scipy.linalg import eigh
    data = open_data()
    SpaceDim = data['SpaceDim']

    if SpaceDim == 1:

        K = data['K']
        M = data['M']

        Kb, Mb = np.array(K.todense()), np.array(M.todense())
        vals, vecs = eigh(Kb[2:, 2:], Mb[2:, 2:])

    elif SpaceDim == 2:

        Mr = data['Mr']
        Kr = data['Kr']
        n_dof_to_reduce = len(data['dof_to_reduce'])

        vals, vecs = eigh(Kr, Mr)
        vecs = np.vstack((np.zeros((n_dof_to_reduce, len(vecs))), vecs))

    data['vecs'] = vecs

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def AssemblyDamping(ζ=0.05, tipo_am="Rayleigh"):
    '''
    Estima la matriz de amortiguamiento según el tipo de método escogido (Rayleigh, ...)
    Estimates the Damping matrix according to the selected method (Rayleigh, ...)

        Input

            K       : Matriz de Rigidez
            K       : Stiffess matrix 
            M       : Matriz de Masas
            M       : Mass Matrix
            ζ       : Fracción del amortiguamiento crítico
            ζ       : Critical damping fraction
            tipo_am : Método para estimar la matriz de amortiguamiento
            tipo_am : Method to estimate the damping matrix
        Output

            C       : Matriz de Amortiguamiento
            C       : Damping Matrix
    '''
    from scipy.linalg import eigh
    import math

    data = open_data()
    elemType = data['elem_type']

    if elemType == 'Q4':

        K = data['Kr']
        M = data['Mr']

    else:
        M = data['M']
        K = data['K']
        Mb = np.array(M.todense())
        Kb = np.array(K.todense())
        M = Mb[2:, 2:]
        K = Kb[2:, 2:]

    vals, vecs = eigh(K, M)
    T = 2 * np.pi / vals

    n = len(M)

    if tipo_am == "Rayleigh":

        tipo = (type(ζ).__name__)

        if tipo == 'float':

            if not 0.0 <= ζ or ζ >= 1.0:

                print("Error! El valor de ζ debe estar en [0,1]")

            ζ = [ζ, ζ]

        elif tipo == 'list':

            if not len(ζ) == 2:

                print("Error! Definir ζ como una lista: [ζi,ζf]")

        else:

            print("Error! ζ debe ser decimal o un lista de 2 valores [ζi,ζf]")

        wi, wf = vals[0]**0.5, vals[int(n / 2)]**0.5

        β = 2 * (wf * ζ[1] - wi * ζ[0]) / (wi**2 + wf**2)

        α = 2 * wi**2**0.5 * ζ[0] - β * wi**2

        print("\nw0,w%s:%s,%s\nα,β=%s,%s" % (int(n / 2), wi, wf, α, β))

        C = α * M + β * K

    else:

        print("Aún no se ha programado para %s" % tipo_am)

    data['C'] = C

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def MDOF_LTH(ug, dt, γ=1 / 2, β=1 / 4, gdl=1):
    ''' 
    Función que estima la respuesta dinámica lineal de una estructura a través del método de newmark usando la formulación incremental.
        Input:
            K:  Matriz de Rigidez
            M:  Matriz de Masas
            C:  Matriz de Amortiguamiento
            ug: Registro de aceleración
            dt: Intervalo de tiempo del registro
            γ,β:Constantes de newmark
            gdl:Grado de Libertad donde actua las aceleraciones
        Output:
            d:  Matriz que contiene los desplazamientos de los gdl en el tiempo
            v:  Matriz que contiene las velocidades de los gdl en el tiempo
            a:  Matriz que contiene las aceleraciones de gdl nodos en el tiempo
        '''
    from numpy.linalg import inv

    data = open_data()
    elemType = data['elem_type']

    if elemType == 'Q4':
        M = np.array(data['Mr'])
        K = np.array(data['Kr'])
    else:
        M = data['M']
        K = data['K']
        Mb = np.array(M.todense())
        Kb = np.array(K.todense())
        M = Mb[2:, 2:]
        K = Kb[2:, 2:]

    C = data['C']

    # Número de grados de libertad
    # Number of degree of freedom

    n = len(M)

    # Número de pasos en el registro
    # Number of steps in the record

    ns = len(ug)

    # Definen matrices que contendrán los desplazamientos, velocidades y aceleraciones de todos los grados de libertad
    # Defines matrices that contain the displacement, velocities, and accelerations of the degrees of freedom.

    dx = np.zeros((ns + 1, n))
    d = np.zeros((ns + 1, n))
    v = np.zeros((ns + 1, n))
    a = np.zeros((ns + 1, n))
    df = np.zeros((ns + 1, n))
    #
    df[0] = 0
    df[1] = -(ug[1] - ug[0])
    #
    c1 = 1 - γ / β
    c2 = γ / (β * dt)
    c3 = dt * (1 - γ / (2 * β))
    #
    a1 = M / (β * dt**2) + γ * C / (β * dt) + K
    a2 = M / (β * dt) + γ * C / β
    a3 = M / (2 * β) - dt * (1 - γ / (2 * β)) * C

    I = np.ones(n) * 0
    I[gdl::2] = 1.0
    MI = M @ I

    # Solución de la ecuación diferencial
    for i in range(1, ns):
        if i % 100 == 0:
            print("Paso:\t%s\nTiempo:\t%s(s)" % (i, i * dt))

        dx[i] = inv(a1) @ (a2 @ v[i - 1] + a3 @ a[i - 1] + MI * df[i])

        #
        d[i] = d[i - 1] + dx[i]
        v[i] = c1 * v[i - 1] + c2 * dx[i] + c3 * a[i - 1]
        a[i] = -inv(M) @ (C @ v[i] + K @ d[i] + MI * ug[i])
        #
        df[i] = -(ug[i] - ug[i - 1])

    # Guardar d, v y a con piclke

    data['d'] = d
    data['v'] = v
    data['a'] = a

    data_file = open('./Data', 'wb')
    pickle.dump(data, data_file)
    data_file.close()


def dinamic_plot(FS=15):

    from matplotlib import animation, rc

    rc('animation', html='jshtml')

    data = open_data()

    NC = data['NC']
    nx = data['nx']
    Connect = data['Connect']
    Nodes = data['Nodes']
    L = data['L']
    H = data['H']
    n_dof_to_reduce = len(data['dof_to_reduce'])
    d = data['d']

    u = np.hstack((np.zeros((d.shape[0], n_dof_to_reduce)), d))

    aux = np.array([])
    aux_1 = np.array([])

    for i in range(NC):

        if (i + 1) % nx == 0 and i != 0:
            for j in range(4):
                aux = np.append(aux, 2 * Connect[i][j])
                aux = np.append(aux, 2 * Connect[i][j] + 1)

            aux = np.append(aux, 2 * Connect[i][0])
            aux = np.append(aux, 2 * Connect[i][0] + 1)
            aux = np.append(aux, 2 * Connect[i + 1 - nx][0])
            aux = np.append(aux, 2 * Connect[i + 1 - nx][0] + 1)
            aux_1 = np.append(aux_1, Connect[i])
            aux_1 = np.append(aux_1, Connect[i][0])
            aux_1 = np.append(aux_1, Connect[i + 1 - nx][0])
        else:
            for j in range(4):

                aux = np.append(aux, 2 * Connect[i][j])
                aux = np.append(aux, 2 * Connect[i][j] + 1)
            aux = np.append(aux, 2 * Connect[i][0])
            aux = np.append(aux, 2 * Connect[i][0] + 1)
            aux_1 = np.append(aux_1, Connect[i])
            aux_1 = np.append(aux_1, Connect[i][0])
    Dq_din = np.zeros((u.shape[0], aux.shape[0]))
    Nodes_din = np.zeros((aux_1.shape[0], Nodes.shape[1]))

    for i, value in enumerate(aux):
        Dq_din[:, i] = u[:, int(value)]

    for i, value in enumerate(aux_1):
        Nodes_din[i] = Nodes[int(value)]

    fig = plt.figure(figsize=(3, 10))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    # line1, = ax.plot([], [], 'r', label="Bern")
    # line2, = ax.plot([], [], 'b--', label="Timo")
    line3, = ax.plot([], [], 'k-', markersize=15, label="Quad")
    # line4, = ax.plot([], [], 'k', label="Lin")
    ax.legend(loc='upper right')
    ax.set_xlim(-10 - L / 2, L / 2 + 10)
    ax.set_ylim(-0.5, H + 0.5)

    def anima(i, factor):
        # line1.set_xdata(np.insert(Db[i][1::2], 0, 0) * factor)
        # line1.set_ydata(xB)
        # line2.set_xdata(np.insert(Dt[i][1::2], 0, 0) * factor)
        # line2.set_ydata(xT)
        X_def, X_incr = deformed_din(Nodes_din, Dq_din[i], factor)
        line3.set_xdata(X_def[:, 0] - L / 2)
        line3.set_ydata(X_def[:, 1])
        # ax.set_title("%5.2f(s)" % (i * 0.02))

        return line3,

    ani = animation.FuncAnimation(fig,
                                  anima,
                                  frames=len(u) - 1,
                                  fargs=(FS, ),
                                  interval=100,
                                  blit=True)
    ani.save('animacion.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def dinamic_plot_1D(label='Bern', FS=15):

    data = open_data()

    d = data['d']
    xB = data['Nodes']
    Len_beam = data['Len_beam']

    fig = plt.figure(figsize=(3, 10))
    ax = fig.add_subplot(111)
    line1, = ax.plot([], [], 'ko-', markersize=5, label=label)
    ax.legend(loc='upper right')
    ax.set_xlim(-Len_beam / 20, Len_beam / 20)
    ax.set_ylim(0, Len_beam + 1)

    #
    def anima(i, factor):
        line1.set_xdata(np.insert(d[i][1::2], 0, 0) * factor)
        line1.set_ydata(xB)
        ax.set_title("%5.2f(s)" % (i * 0.02))
        return line1,

    #
    ani = animation.FuncAnimation(fig,
                                  anima,
                                  frames=len(d) - 1,
                                  fargs=(FS, ),
                                  interval=100,
                                  blit=True)
    ani.save('animacion.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


import subprocess

# import ffmpeg


def get_codecs():
    cmd = "ffmpeg -codecs"
    x = subprocess.check_output(cmd, shell=True)
    x = x.split(b'\n')
    for e in x:
        print(e)


def get_formats():
    cmd = "ffmpeg -formats"
    x = subprocess.check_output(cmd, shell=True)
    x = x.split(b'\n')
    for e in x:
        print(e)


def convert_seq_to_mov():

    input = r"C:\Users\RICK\Desktop\Nueva carpeta\Nueva carpeta\Model_Deformada_TH_%05d.png"
    # input = r"C:\Users\HP\Desktop\FFMPEG\smoke\dense_smoke_p001.%03d.png"

    output = r"C:\Users\RICK\Desktop\Nueva carpeta\out.mp4"

    frame_rate = 5

    cmd = f'ffmpeg -framerate {frame_rate} -i "{input}" "{output}"'

    print(cmd)

    subprocess.check_output(cmd, shell=True)
    # subprocess.run(f'cmd /k ffmpeg', shell=True)


# get_codecs()
# get_formats()
# convert_seq_to_mov()

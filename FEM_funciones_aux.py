# Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe)
# Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
# Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##
# Este código se basó en las rutinas de matlab FEMCode
# realizado inicialmente por Garth N .Wells (2005)
# para la clase CT5123 en TU Delft, Países Bajos
##

# from _typeshed import Self
import matplotlib.tri as tri
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
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
    with open('./data/Data.sav', 'rb') as f:
        data = pickle.load(f)
    return data


def ProblemData(SpaceDim, pde):

    try:
        os.mkdir('data')
    except:
        pass

    data = {}
    data['SpaceDim'] = SpaceDim
    data['pde'] = pde

    with open('./data/Data.sav', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def ElementData(dof, nodes, noInt, type):

    data = open_data()
    # Agragar a Data los datos de los elementos
    data['elem_dof'] = dof
    data['elem_nodes'] = nodes
    data['elem_noInt'] = noInt
    data['elem_type'] = type

    with open('./data/Data.sav', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def MassData(dof, nodes, noInt, type):

    data = open_data()
    # Agragar a Data los datos de los elementos
    data['mass_dof'] = dof
    data['mass_nodes'] = nodes
    data['mass_noInt'] = noInt
    data['mass_type'] = type

    with open('./data/Data.sav', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def ModelData(E, v, thickness, density, selfWeight, gravity):

    data = open_data()
    # Agragar a Data los datos del modelo
    data['E'] = E
    data['v'] = v
    data['thickness'] = thickness
    data['density'] = density
    data['selfWeight'] = selfWeight
    data['gravity'] = gravity

    with open('./data/Data.sav', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def GenQuadMesh(L, H, lc, fd="./INPUT"):
    '''Función que crea el mesh de un elemento rectangular usando
      elementos Quad (Crea un archivo .msh).
    '''
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("demo")
    ##
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(L, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(0, H, 0, lc, 3)
    gmsh.model.geo.addPoint(L, H, 0, lc, 4)
    ##
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(1, 3, 4)
    ##
    gmsh.model.geo.addCurveLoop([1, 2, 3, -4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    ##
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    ##
    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setRecombine(2, 1)
    ##
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "quad")
    ##
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(fd + "/rectangle_%4.2f.msh" % lc)
    # gmsh.fltk.run() ##Abre el gmsh
    gmsh.finalize()
    return fd + "/rectangle_%4.2f.msh" % lc


def GenQuadMesh_2D(L, H, lc):
    '''Función que crea el mesh de un elemento rectangular usando
    códigos programados en Python
    '''
    data = open_data()
    SpaceDim = data['SpaceDim']

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

    print("nx = %d, ny = %d" % (nx, ny))

    ni = 0
    noNodes = (nx + 1) * (ny + 1)
    x = np.zeros((noNodes, 3), dtype=np.float64)

    for j in range(ny + 1):
        for i in range(nx + 1):
            x[ni] = (ms_x * i, ms_y * j, 0)
            ni = ni + 1

    noElements = nx * ny
    connect = np.zeros((noElements, 4), dtype=np.int32)

    k = 0
    for i in range(0, ny):
        for j in range(0, nx):
            connect[k, 0] = j + ((i) * (nx + 1))
            connect[k, 1] = j + ((i) * (nx + 1)) + 1
            connect[k, 2] = j + ((i + 1) * (nx + 1)) + 1
            connect[k, 3] = j + ((i + 1) * (nx + 1))
            k = k + 1

    # ------- Clean up and close -----------------------
    # Delete coordinates
    if (SpaceDim == 1):
        x[:, 1:3] = 0.0
    elif (SpaceDim == 2):
        x = np.delete(x, 2, 1)

    # class Mesh:
    # 	NN = noNodes
    # 	NC = noElements
    # 	Nodos = x.T
    # 	Conex = connect.T

    data['NN'] = noNodes
    data['NC'] = noElements
    data['Nodos'] = x.T
    data['Conex'] = connect.T

    with open('./data/Data.sav', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def gmsh_read(msh_file, ProblemData, ElementData):
    '''Función que retorna los nodos y conexiones al leer el archivo .msh
      realizado por gmsh (función adaptada del código en matlab FEMCode).
    '''
    print('Leyendo archivo msh')
    file_input = open(msh_file, 'rt')
    # ------------- Test input file ----------------------
    # Read 2 first lines and check if we have mesh format 2
    mesh_format = file_input.readline()[:-1]
    line = file_input.readline()
    fmt = int(float(line.split()[0]))
    if (mesh_format != '$MeshFormat' or fmt != 2):
        print('The mesh format is NOT version 2!')

# -------------------- Nodes  --------------------------
# Process file until we get the line with the number of nodes
    buf = file_input.readline()[:-1]
    while ('$Nodes' != buf):
        buf = file_input.readline()[:-1]
    noNodes = int(file_input.readline()[:-1])  # Extract number of nodes
    # Initialise nodes matrix [x1, y1, z1 x2, y2, z2 .... xn, yn, zn]
    x = np.zeros((noNodes, 3), dtype=np.float64)
    for i in range(noNodes):  # Get nodal coordinates
        buf = [float(y) for y in file_input.readline()[:-1].split()]
        x[i] = buf[1:4]  # we throw away the node numbers!

# ------------ Elements --------------------
# Process file until we get the line with the number of elements
    while ('$Elements' != buf):
        buf = file_input.readline()[:-1]
# Extract number of elements
    noElements = int(file_input.readline()[:-1])
    # Get first line of connectivity
    buf = [int(y) for y in file_input.readline()[:-1].split()]
    # Number of nodes per element
    no_nodes_per_elem = len(buf) - (3 + buf[2])
    tipo = buf[1]  # Get type of element
    # Verify that we have the correct element
    if (no_nodes_per_elem != ElementData.nodes):  # Check number of nodes
        print(
            'The number of nodes per element in the mesh differ from ElementData.nodes'
        )
# Check element type (gmsh 2.0 manual )
    if (ElementData.type == 'Tri3'):
        if (tipo != 2):
            print('Element type is not Tri3')
    elif (ElementData.type == 'Quad4'):
        if (tipo != 3):
            print('Element type is not Quad4')
    elif (ElementData.type == 'Tri6'):
        if (tipo != 9):
            print('Element type is not Tri6')
    elif (ElementData.type == 'Tet4'):
        if (tipo != 4):
            print('Element type is not Tet4')
    else:  # Default error message
        print('Element type %s is not supported', ElementData.type)

# --------- Initialise connecticity matrix and write first line ------------
    connect = np.zeros((noElements, no_nodes_per_elem), dtype=np.int32)
    connect[0, :] = buf[3 + buf[2]:len(buf)]
    # Get element connectivity
    # FIXME: check that the nodes on the elements are numbered correctly!
    for i in range(1, noElements):
        buf = [int(y) for y in file_input.readline()[:-1].split()]
        # Only one type of elements is allowed in the mesh
        if (tipo != buf[1]):
            print(
                'More than one type of elements is present in the mesh, did you save all elements?'
            )
        # throw away element number, type and arg list
        connect[i, :] = buf[3 + buf[2]:len(buf)]


# ------- Clean up and close -----------------------
# Delete coordinates
    if (ProblemData.SpaceDim == 1):
        x[:, 1:3] = 0.0
    elif (ProblemData.SpaceDim == 2):
        x = np.delete(x, 2, 1)
    file_input.close()  # Close file

    # ------- Add members to object  ---------------
    class Mesh:
        NN = noNodes
        NC = noElements
        Nodos = x.T
        Conex = connect.T - 1

    return Mesh


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


def genBC_2D(lim=0.01):
    ''' Función que aplica las condiciones de borde en un punto especificando
        las coordenadas donde se encuentra. (Utiliza un radio de búsqueda)
    '''

    data = open_data()
    X = data['Nodos'].T
    BC_coord = data['BC_coord']

    NN, k = len(X), 0
    BC_data = np.zeros((len(BC_coord), 4))
    BC_data[:, 1:] = BC_coord[:, 2:]
    for x, y in BC_coord[:, 0:2]:
        for i in range(NN):
            if i == 0:
                er0, ind = ((X[i, 0] - x)**2 + (X[i, 1] - y)**2)**0.5, 0
                continue
            er = ((X[i, 0] - x)**2 + (X[i, 1] - y)**2)**0.5
            if er < er0:
                er0, ind = er, i
                continue
            ##
        if er0 < lim:
            BC_data[k, 0] = ind + 1
            k = k + 1
        else:
            print(
                "No se encuentra un punto cerca a (%s,%s) en un radio de %s" %
                (x, y, lim))
            k = k + 1

    data['BC_data'] = BC_data

    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def fix_nodes_rec(dir, dist, gdl, lim):
    '''Función que retorna los nodos que se encuentran en una dirección
    especificada.
    '''

    data = open_data()
    X = data['Nodos'].T

    dic_vector = {}
    if dir == 'x':

        for i in gdl:
            aux = np.where(abs(X[:, 1] - dist) < lim)
            vector = np.hstack((X[aux], [[1, i, 0]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'y':

        for i in gdl:
            aux = np.where(abs(X[:, 0] - dist) < lim)
            vector = np.hstack((X[aux], [[1, i, 0]] * len(aux[0])))
            dic_vector.update({i: vector})

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_coord' in data.keys():
        data['BC_coord'] = np.vstack((data['BC_coord'], vector))
    else:
        data['BC_coord'] = vector

    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def fix_node(x, y, gdl, lim):
    '''Función que retorna los nodos que se encuentran en una dirección
    especificada.
    '''

    data = open_data()
    X = data['Nodos'].T
    dic_vector = {}

    for i in gdl:

        aux_1 = np.where(abs(X[:, 0] - x) < lim)
        aux_2 = np.where(abs(X[:, 1] - y) < lim)
        aux = np.intersect1d(aux_1, aux_2)

        vector = np.hstack((X[aux], [[1, i, 0]] * len(aux)))
        dic_vector.update({i: vector})

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_coord' in data.keys():
        data['BC_coord'] = np.vstack((data['BC_coord'], vector))
    else:
        data['BC_coord'] = vector

    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def force_dist_rec(dir, dist, force, gdl, lim):
    '''Función que retorna los nodos que se encuentran en una dirección
    especificada.
    '''

    data = open_data()
    X = data['Nodos'].T

    dic_vector = {}
    if dir == 'x':

        for i in gdl:
            aux = np.where(abs(X[:, 1] - dist) < lim)
            vector = np.hstack((X[aux], [[0, i, force]] * len(aux[0])))
            dic_vector.update({i: vector})

    elif dir == 'y':

        for i in gdl:
            aux = np.where(abs(X[:, 0] - dist) < lim)
            vector = np.hstack((X[aux], [[0, i, force]] * len(aux[0])))
            dic_vector.update({i: vector})

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_coord' in data.keys():
        data['BC_coord'] = np.vstack((data['BC_coord'], vector))
    else:
        data['BC_coord'] = vector

    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def force_node(x, y, force, gdl, lim):
    '''Función que retorna los nodos que se encuentran en una dirección
    especificada.
    '''

    data = open_data()
    X = data['Nodos'].T
    dic_vector = {}

    for i in gdl:
        print
        aux_1 = np.where(abs(X[:, 0] - x) < lim)
        aux_2 = np.where(abs(X[:, 1] - y) < lim)

        aux = np.intersect1d(aux_1, aux_2)

        vector = np.hstack((X[aux], [[0, i, force]] * len(aux)))
        dic_vector.update({i: vector})

    vector = np.vstack(tuple(dic_vector.values()))

    if 'BC_coord' in data.keys():
        data['BC_coord'] = np.vstack((data['BC_coord'], vector))
    else:
        data['BC_coord'] = vector

    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def CuadraturaGauss(puntos, dim):
    '''
    Esta función define los puntos de integración según la
    cuadratura de Gauss
    ---------------
    Input:
       puntos:    El número de puntos de integración de Gauss
       dim:         Dimensión del elemento finito
    Output:
       gp:           Arreglo cuya primera fila son los pesos y las
                      demás las pocisiones respectivas
    '''
    if dim == 1:
        if puntos == 1:  # Integración de Gauss de 1 punto en 1D
            gp = np.zeros((2, 1))
            gp[0], gp[1] = 2.0, 0.0

        elif puntos == 2:  # Integración de Gauss de 2 puntos en 1D
            gp = np.zeros((2, 2))
            gp[0, 0], gp[1, 0] = 1.0, -1 / 3**0.5
            gp[0, 1], gp[1, 1] = 1.0, 1 / 3**0.5

        elif puntos == 4:
            gp = np.zeros((2, 4))
            a, b = 0.33998104358484, 0.8611363115941
            wa, wb = 0.65214515486256, 0.34785484513744
            gp[0, 0], gp[1, 0] = wb, -b
            gp[0, 1], gp[1, 1] = wa, -a
            gp[0, 2], gp[1, 2] = wa, a
            gp[0, 3], gp[1, 3] = wb, b
        else:
            print("Debes programarpar %s puntos aún" % puntos)
    ##
    elif dim == 2:
        if puntos == 1:  # Integración de Gauss de 1 punto para Tri3
            gp = np.zeros((3, 1))
            gp[0] = 1.0 * 0.5  # peso*porcentaje del Jacobiano
            gp[1], gp[2] = 1.0 / 3.0, 1.0 / 3.0  # Coordenadas de Integración
            #
        elif puntos == 4:  # Integración de Gauss de 2x2 en 2D para Quad
            gp = np.zeros((3, 4))
            gp[0, :] = 1.0
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1, 3] = -1.0 / \
                3**0.5, 1.0/3**0.5, 1.0/3**0.5, -1.0/3**0.5
            gp[2, 0], gp[2, 1], gp[2, 2], gp[2, 3] = -1.0 / \
                3**0.5, -1.0/3**0.5, 1.0/3**0.5, 1.0/3**0.5
        else:
            print("Debes programarpar %s puntos aún" % puntos)
    ##
    else:
        print("Debes programar para %sD aún" % dim)
    return gp.T


def FunciónForma(X, gp, tipo):
    '''
    Esta función define funciones de forma y sus derivadas
    en coordenadas naturales para un elemento finito de
    coordenadas X, N(X)=N(xi),dN(X)=dN(xi)*J-1
    -----------------
    Input:
       X:             Matriz de coordenadas del elemento
       gp:            Arreglo que contiene los parametros de la
                        cuadratura de Gauss
       tipo:          Tipo de elemento finito
    Output:
       N:             Matriz de funciones de Forma
       dN:           Matriz de la derivada de las funciones de Forma
       ddN:         Matriz de la segunda derivada de las funciones de Forma
       j:               Jacobiano para realizar la integración usando el mapeo isoparampetrico
    '''
    if tipo == 'Bar1':
        N, dN, J = np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((1, 1))
        #
        xi = gp[1]
        N[0], N[1] = -xi / 2 + 0.5, xi / 2 + 0.5
        dN[0, 0], dN[1, 0] = -1 / 2, 1 / 2
        #
    elif tipo == 'BarB':
        N, dN, ddN, J = np.zeros((4, 1)), np.zeros((4, 1)), np.zeros(
            (4, 1)), np.zeros((1, 1))
        #
        xi = gp[1]
        le = X[1] - X[0]
        N[0] = le * (1 - xi)**2 * (1 + xi) / 8
        N[1] = (1 - xi)**2 * (2 + xi) / 4
        N[2] = le * (1 + xi)**2 * (-1 + xi) / 8
        N[3] = (1 + xi)**2 * (2 - xi) / 4
        dN[0] = -(1 - xi) * (3 * xi + 1) / 4
        dN[1] = -3 * (1 - xi) * (1 + xi) / (2 * le)
        dN[2] = (1 + xi) * (3 * xi - 1) / 4
        dN[3] = 3 * (1 - xi) * (1 + xi) / (2 * le)
        ddN[0] = (3 * xi - 1) / le
        ddN[1] = 6 * xi / (le**2)
        ddN[2] = (3 * xi + 1) / le
        ddN[3] = -6 * xi / (le**2)
        j = le / 2
        return N, dN, ddN, j
        #
    elif tipo == 'Tri3':
        N, dN, J = np.zeros((3, 1)), np.zeros((3, 2)), np.zeros((2, 2))
        #
        xi, eta = gp[1], gp[2]
        N[0], N[1], N[2] = xi, eta, 1.0 - xi - eta
        dN[0, 0], dN[1, 0], dN[2, 0] = 1.0, 0.0, -1.0  # dN/d(xi)
        dN[0, 1], dN[1, 1], dN[2, 1] = 0.0, 1.0, -1.0  # dN/d(eta)
        #
    elif tipo == 'Quad4':
        N, dN, J = np.zeros((4, 1)), np.zeros((4, 2)), np.zeros((2, 2))
        a = np.array([-1.0, 1.0, 1.0, -1.0])  # coordenadas x de los nodos
        b = np.array([-1.0, -1.0, 1.0, 1.0])  # coordenadas y de los nodos
        xi = gp[1]
        eta = gp[2]
        ##
        N = 0.25 * (1.0 + a[:] * xi + b[:] * eta + a[:] * b[:] * xi * eta)
        dN[:, 0] = 0.25 * (a[:] + a[:] * b[:] * eta)
        dN[:, 1] = 0.25 * (b[:] + a[:] * b[:] * xi)
        #
    else:
        print("Debes programar para el tipo %s aún" % tipo)
    # Calculamos la matriz jacobiana y su determinante
    J = X @ dN
    ##
    if len(J) > 1:
        j = det(J)
        dN = dN @ inv(J)
    else:
        j = J[0]
        dN = dN / j
    if (j < 0):
        print("Cuidado: El jacobiano es negativo!")
        # print(X,'\n',dN,'\n',J,'\n',j)
    ddN = 0.0  # Retorna 0.0 cuando no es Bernoulli
    return N, dN, ddN, j

    # Ejemplo de aplicacion de las funciones de forma para Tri3
    # X=Mesh.Nodos[:,Mesh.Conex[:,0]]#np.array([[3,2],[8,7],[1,12]]).T
    # gp=CuadraturaGauss(1,2)
    # [N,dN,ddN,j]=FunciónForma(X,gp.T,tipo='Tri3')


def Bernoulli(A, x, N, dN, ddN, ProblemData, ElementData, ModelData, dX, tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    '''
    n = ElementData.nodes
    m = ElementData.dof
    EI = ModelData.EI
    Area = ModelData.Area
    ##
    if tipo == 'MatrizK':
        A = EI * ddN @ ddN.T
        A = A * dX

    elif tipo == 'MasaConsistente':
        rho = ModelData.density
        A = rho * N @ N.T * dX * Area
        ##
    elif tipo == 'MasaConcentrada':
        rho = ModelData.density
        B = rho * N @ N.T * dX * Area
        ##
        one = np.zeros(m * n)
        one[:] = 1.0
        B = B @ one
        A = np.zeros((m * n, m * n))
        # Concentrando Masas (Mejorar proceso si |A| <= 0)
        for i in range(m * n):
            A[i, i] = B[i]
        ##
    elif (tipo == 'VectorF'):
        # Formando matriz N para v
        N_v = np.zeros((m * n, 1))
        N_v = N
        A = np.zeros((m * n, 1))
        f = ModelData.fy
        A = N_v.T * f * dX
        ##
    else:
        print("Debes programar para el tipo %s aún" % tipo)
    return A


def Timoshenko(A, x, N, dN, ddN, ProblemData, ElementData, ModelData, dX,
               tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    '''
    n = ElementData.nodes
    m = ElementData.dof
    EI = ModelData.EI
    GAs = ModelData.GAs
    Area = ModelData.Area
    ##
    if tipo == 'MatrizK':
        if (ProblemData.SpaceDim == 1):
            # Formando N para theta
            N_theta = np.zeros((1, m * n))
            N_theta[0, 0::m] = N[:, 0].T
            # Formando N para v
            N_v = np.zeros((1, m * n))
            N_v[0, 1::m] = N[:, 0].T
            # Formando B para theta
            B_theta = np.zeros((1, m * n))
            B_theta[0, 0::m] = dN[:, 0].T
            # Formando B para for v
            B_v = np.zeros((1, m * n))
            B_v[0, 1::m] = dN[:, 0].T
            ##
        elif (ProblemData.SpaceDim == 2):
            print('Solo es válido para 1D')
        elif (ProblemData.SpaceDim == 3):
            print('Solo es válido para 1D')
        ##
        # Calculando Matriz
        A = B_theta.T * EI @ B_theta + N_theta.T * GAs @ N_theta - N_theta.T * GAs @ B_v
        A = A - B_v.T * GAs @ N_theta + B_v.T * GAs @ B_v
        A = A * dX
    ##
    elif tipo == 'MasaConsistente':
        rho = ModelData.density
        nulo = np.array([0])
        N = np.array([nulo, N[0], nulo, N[1]])
        A = rho * N @ N.T * dX * Area
        # Artificio para no obetener una Matriz singular
        A[0, 0] = A[0, 0] + 0.01 * A[1, 1]
        A[2, 2] = A[2, 2] + 0.01 * A[3, 3]
    ##
    elif tipo == 'MasaConcentrada':
        rho = ModelData.density
        nulo = np.array([0])
        N = np.array([nulo, N[0], nulo, N[1]])
        B = rho * N @ N.T * dX * Area
        # Artificio para no obetener una Matriz singular
        B[0, 0] = B[0, 0] + 0.01 * B[1, 1]
        B[2, 2] = B[2, 2] + 0.01 * B[3, 3]
        ##
        one = np.zeros(m * n)
        one[:] = 1.0
        B = B @ one
        A = np.zeros((m * n, m * n))
        # Concentrando Masas
        for i in range(m * n):
            A[i, i] = B[i]

    elif (tipo == 'VectorF'):
        # Formando matriz N para theta
        N_theta = np.zeros((1, m * n))
        N_theta[0, 0::m] = N[:, 0].T
        # Formando matriz N para v
        N_v = np.zeros((1, m * n))
        N_v[0, 1::m] = N[:, 0].T
        #
        f = ModelData.fy
        A = N_v * f * dX

    else:
        print("Debes programar para el tipo %s aún" % tipo)
    return A


def Elasticidad(A, X, N, dN, dNN, dX, tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    '''

    data = open_data()

    n = data['elem_nodes']
    m = data['elem_dof']
    t = data['thickness']

    SpaceDim = data['SpaceDim']
    density = data['density']
    selfweight = data['selfWeight']
    gravity = data['gravity']

    if tipo == 'MatrizK':

        E, v = data['E'], data['v']  # Estado plano de esfuerzos
        # E,v = E/(1.0-v*2),v/(1-v)## Estado plano de deformaciones
        if SpaceDim == 2:
            # Formando Matriz D
            D = np.zeros((3, 3))
            D[0, 0], D[1, 1] = 1.0, 1.0
            D[0, 1], D[1, 0] = v, v
            D[2, 2] = 0.5 * (1.0 - v)
            D = E * D / (1 - v**2)
            # print("D=",D)
            # Formando Matriz B
            B = np.zeros((3, m * n))
            for i in range(m):
                B[i, i::m] = dN[:, i]
            B[2, 0::m] = dN[:, 1]
            B[2, 1::m] = dN[:, 0]
        else:
            print("Debes programar para %sD aún" % dim)
        #
        A = B.T @ D @ B * dX * t

    elif tipo == 'MasaConsistente':
        Nmat = np.zeros((m, m * n))
        rho = density
        for i in range(m):
            Nmat[i, i::m] = N[:].T
        #####
        A = rho * Nmat.T @ Nmat * dX * t

    elif tipo == 'MasaConcentrada':
        Nmat = np.zeros((m, m * n))
        rho = density
        for i in range(m):
            Nmat[i, i::m] = N[:].T
        ####
        B = rho * Nmat.T @ Nmat * dX * t
        one = np.zeros(m * n)
        one[:] = 1.0
        B = B @ one
        A = np.zeros((m * n, m * n))
        # Concentrando Masas
        for i in range(m * n):
            A[i, i] = B[i]

    elif tipo == 'VectorF':
        # Formando Matriz F
        Nmat = np.zeros((m, m * n))
        for i in range(m):
            Nmat[i, i::m] = N[:].T
        f = selfweight * gravity[0:m]
        ##
        A = Nmat.T @ f * dX * t
    else:
        print("Debes programar para el tipo %s aún" % tipo)
    return A


def DofMap(DofNode, connect, NodesElement):
    '''Función que mapea los grados de libertad correspondientes a un EF
    '''
    dof = []
    for i in range(NodesElement):
        for j in range(DofNode):
            dof.append(DofNode * (connect[i]) + j)
    return dof


def AssembleMatrix(MatrixType):
    '''Función que realiza el ensamble de la matriz K, de Ku=F
    '''
    data = open_data()

    NN = data['NN']
    elem_dof = data['elem_dof']
    elem_type = data['elem_type']
    elem_noInt = data['elem_noInt']
    elem_nodes = data['elem_nodes']
    Nodos = data['Nodos']
    SpaceDim = data['SpaceDim']
    Conex = data['Conex']
    pde = data['pde']

    N = NN * elem_dof
    n = elem_nodes * elem_dof
    # Definiendo matriz sparse para la matriz global
    A = sparse.lil_matrix((N, N), dtype=np.float64)
    # Definiendo Matriz para elementos
    A_e = np.zeros(n, dtype=np.float64)
    A_int = np.zeros(n, dtype=np.float64)
    # Se obtiene los pesos y posiciones de la Cuadratura de Gauss
    gp = CuadraturaGauss(elem_noInt, SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    for element in Conex.T:
        # Asigando coordenadas de nodos y connectividad
        connect_element = element[0:elem_nodes]
        if SpaceDim == 1:
            x_element = Nodos[connect_element]
        else:
            x_element = Nodos[:, connect_element]
        # Bucle para realizar la integración según Cuadratura de Gauss
        for gauss_point in gp:
            # Se calcula Las Funciones de Forma
            [N, dN, ddN, j] = FunciónForma(x_element, gauss_point, elem_type)
            # Se calcula la Matriz de cada Elemento
            dX = gauss_point[0] * j
            # Evaluar el PDE (Elasticidad, Timoshenko, Bernoulli, etc)
            A_int = eval(pde + '(A_int, x_element, N, dN,ddN, dX, MatrixType)')
            A_e = A_e + A_int
        # if MatrixType=="MatrizK": print("K_elemento",A_e)
        # Se mapea los grados de libertad
        dof = DofMap(elem_dof, connect_element, elem_nodes)
        # Ensamblando
        cont = 0
        for k in dof:
            A[k, dof] = A[k, dof] + A_e[cont]
            cont = cont + 1
        # Se resetea la Matriz del Elemento
        A_e[:] = 0.0

    if MatrixType == "MatrizK":
        data['K'] = A

        with open('data/Data.sav', 'wb') as f:
            pickle.dump(data, f)

    elif MatrixType == "MasaConcentrada":
        data['M'] = A
        with open('data/Data.sav', 'wb') as f:
            pickle.dump(data, f)


def AssembleVector(MatrixType):
    '''Función que realiza el ensamble del vector F, de Ku=F
    '''

    data = open_data()

    NN = data['NN']
    elem_dof = data['elem_dof']
    elem_type = data['elem_type']
    elem_noInt = data['elem_noInt']
    Nodos = data['Nodos']
    SpaceDim = data['SpaceDim']
    Conex = data['Conex']
    elem_nodes = data['elem_nodes']
    pde = data['pde']

    N = NN * elem_dof
    n = elem_nodes * elem_dof

    # Definiendo vector f global
    f = np.zeros(N, np.float64)
    # Definiendo Matriz para elementos
    f_e = np.zeros(n, np.float64)
    f_int = np.zeros(n, np.float64)
    # Se obtiene los pesos y posiciones de la Cuadratura de Gauss
    gp = CuadraturaGauss(elem_noInt, SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    for element in Conex.T:
        # Asigando coordenadas de nodos y connectividad
        connect_element = element[0:elem_nodes]
        if SpaceDim == 1:
            x_element = Nodos[connect_element]
        else:
            x_element = Nodos[:, connect_element]
        # Bucle para realizar la integración según Cuadratura de Gauss
        for gauss_point in gp:
            # Se calcula Las Funciones de Forma
            [N, dN, ddN, j] = FunciónForma(x_element, gauss_point, elem_type)
            # Se calcula la Matriz de cada Elemento
            dX = gauss_point[0] * j
            # Evaluar el PDE (Elasticidad, Timoshenko, Bernoulli, etc)
            f_int = eval(pde + '(f_int, x_element, N, dN, ddN, dX, "VectorF")')
            f_e = f_e + f_int
        # Se mapea los grados de libertad
        dof = DofMap(elem_dof, connect_element, elem_nodes)
        # Ensamblando
        # print(dof,f_e)
        f[dof] = f[dof] + f_e
        f_e = 0.0

    data['f'] = f
    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def ApplyBC():
    '''Esta función aplica las condiciones de borde especificadas
    '''
    data = open_data()
    elem_dof = data['elem_dof']
    BC_data = data['BC_data']
    A = data['K']
    f = data['f']

    for bc in BC_data:
        if int(bc[1]) == 0:  # Neumann
            dof = int(elem_dof * (bc[0] - 1) + bc[2]) - 1
            print("Neumann, DOF:", dof)
            f[dof] = f[dof] + bc[3]
        elif int(bc[1]) == 1:  # Dirichlet
            dof = int(elem_dof * (bc[0] - 1) + bc[2]) - 1
            print("Dirichlet, DOF:", dof)
            A[dof, :] = 0.0
            A[dof, dof] = 1.0
            f[dof] = bc[3]
        else:
            print('Condición de Borde Desconocida')

    data['K'] = A
    data['f'] = f
    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


def Analysis():
    data = open_data()
    K = data['K']
    f = data['f']

    u = spsolve(K.tocsr(), f)

    data['u'] = u
    with open('data/Data.sav', 'wb') as f:
        pickle.dump(data, f)


########## Funciones Diversas ############


def Deformada(u, FS=1 / 500):
    ''' Función que agrega una deformación a la pocisión de Nodos
        Input:
            X:      Arreglo que contiene las coordenadas de todos los Nodos
            u:      Arreglo que contiene las deformaciones calculadas con el MEF
            FS:     Factor de Escala
        Output:
            X_def:  Arreglo que contiene la deformada del sistema.
        '''
    data = open_data()
    X = data['Nodos'].T
    NN = len(X)
    print(NN)
    X_incr = np.zeros(X.shape)
    X_def = np.zeros(X.shape)

    for i in range(NN):
        X_incr[i] = FS * np.array([u[2 * i], u[2 * i + 1]])
        X_def[i] = X[i] + FS * np.array([u[2 * i], u[2 * i + 1]])

    return X_def, X_incr


def tridiag(a=2.1, n=5):
    ''' Función que retorna una matriz de rigidez uniforme de a para n gdl
    '''
    aa = [-a for i in range(n - 1)]
    bb = [2 * a for i in range(n)]
    bb[-1] = a
    cc = [-a for i in range(n - 1)]
    return np.diag(aa, -1) + np.diag(bb, 0) + np.diag(cc, 1)


def K_reductor(K, dof):
    '''Función que elimina grados de libertad que no se desea analizar
    '''
    k = 0
    for i in dof:
        K = np.delete(np.delete(K, i - k, 0), i - k, 1)
        k = k + 1


# print(i,len(K))
    return K


def V_insertor(V, dof):
    '''Función que agrega valores nulos aun vector en posiciones especificadas
    '''
    for i in dof:
        V = np.insert(V, i, 0)
    return V


# converts quad elements into tri elements
def quads_to_tris(quads):
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


# plots a finite element mesh


def plot_fem_mesh(nodes_x, nodes_y, elements):
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.2)


def plot_model_mesh():

    data = open_data()

    NC = data['NC']
    Nodos = data['Nodos'].T
    Conex = data['Conex'].T

    ##
    import matplotlib.tri as tri
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    elements_quads = np.zeros((NC, 4), dtype=int)

    # FEM data
    nodes_x = Nodos[:, 0]
    nodes_y = Nodos[:, 1]
    # elements = elements_quads
    elements = Conex

    # plot the finite element mesh
    plot_fem_mesh(nodes_x, nodes_y, elements)

    plt.axis('equal')
    plt.show()
    # plt.savefig('Mesh.png')
    plt.close()


def plot_model_deformada(dir, FS):

    ##
    import matplotlib.tri as tri
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    ##
    data = open_data()

    Nodos = data['Nodos'].T
    NC = data['NC']
    Conex = data['Conex'].T
    u = data['u']

    X_Def, X_incr = Deformada(u, FS)

    # FEM data
    nodes_x = X_Def[:, 0]
    nodes_y = X_Def[:, 1]

    if dir == 'x':
        nodal_values = X_incr[:, 0]
    else:
        nodal_values = X_incr[:, 1]

    elements_quads = Conex

    elements_all_tris = quads_to_tris(elements_quads)

    # create an unstructured triangular grid instance
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)

    # plot the finite element mesh
    plot_fem_mesh(nodes_x, nodes_y, elements_quads)

    # plot the contours
    plt.tricontourf(triangulation, nodal_values, cmap="RdBu_r")

    norm = colors.Normalize(vmin=min(nodal_values / FS),
                            vmax=max(nodal_values / FS))

    # Ocultar los valores de los ejes
    # plt.axis('off')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                 orientation='horizontal',
                 label='Deformación (m)')
    plt.axis('equal')
    # plt.savefig('Model_Deformada.png')
    plt.show()
    plt.close()

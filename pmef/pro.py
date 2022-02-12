from numpy import zeros, array
from numpy.linalg import det, inv
from scipy.sparse import lil_matrix

def AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, MatrixType):
    '''Función que realiza el ensamble de la matriz K, de Ku = F
    Input:
            Mesh:       Clase que contiene los nodos y conexiones
                        obtenidos del mallado.
            __Data:     Información del EF, problema a resolver y del
                        modelo utilizado.
            MatrixType: Texto que indica que arreglo se quiere obtener
    Output:
            A:          Arreglo obtenido del ensamble de las matrices de
                        los elementos finitos.
    '''
    N = Mesh.NN*ElementData.dof
    n = ElementData.nodes*ElementData.dof
    # Define matriz sparse para la matriz global 
    A = lil_matrix((N,N), dtype='float64')
    # Define Matriz para elementos
    A_e   = zeros(n, dtype='float64')
    A_int = zeros(n, dtype='float64')
    # Obtiene pesos y posiciones de la Cuadratura de Gauss
    gp = CuadraturaGauss(ElementData.noInt, ProblemData.SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    for connect_element in Mesh.Conex:
        # Obtiene coordenadas de nodos del elemento
        x_element = Mesh.Nodos[connect_element]
        # Bucle para realizar la integración según Cuadratura de Gauss
        for gauss_point in gp:
            # Evalua los puntos de Gauss en las Funciones de Forma
            [N, dN,ddN, j] = FunciónForma(x_element, gauss_point, ElementData.type)
            dX = gauss_point[0]*j
            # Obtiene la Matriz de cada Elemento según el Problema(Elasticidad, Timoshenko, Bernoulli, etc)
            A_int = eval(ProblemData.pde +'(A_int, x_element, N, dN,ddN, ProblemData,ElementData, ModelData, dX, MatrixType)')
            A_e = A_e + A_int
        # Mapea los grados de libertad
        dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
        # Ensambla la matriz del elemento en la matriz global
        cont=0
        for k in dof:
            A[k,dof]=A[k,dof]+A_e[cont]
            cont=cont+1
        # Asigna 0 a la Matriz del Elemento
        A_e[:] = 0.0
    return A
    
def AssembleVector(Mesh, ElementData, ProblemData, ModelData, MatrixType):
    '''Función que realiza el ensamble del vector F, de Ku=F
    Input:
            Mesh:       Clase que contiene los nodos y conexiones
                        obtenidos del mallado.
            __Data:     Información del EF, problema a resolver y del
                        modelo utilizado.
            MatrixType: Texto que indica que arreglo se quiere obtener
    Output:
            f:          Vector obtenido del ensamble de los vectores f_e
                        de los elementos finitos.
    '''
    N = Mesh.NN*ElementData.dof
    n = ElementData.nodes*ElementData.dof
    # Definiendo vector f global 
    f = zeros(N,'float64')
    # Definiendo Matriz para elementos
    f_e    = zeros(n,'float64')
    f_int  = zeros(n,'float64')
    # Se obtiene los pesos y posiciones de la Cuadratura de Gauss
    gp = CuadraturaGauss(ElementData.noInt,ProblemData.SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    for connect_element in Mesh.Conex:
        # Asigando coordenadas de nodos y connectividad
        x_element = Mesh.Nodos[connect_element]
        # Bucle para realizar la integración según Cuadratura de Gauss
        for gauss_point in gp:
            # Se calcula Las Funciones de Forma
            [N, dN,ddN, j] = FunciónForma(x_element, gauss_point,ElementData.type)      
            # Se calcula la Matriz de cada Elemento
            dX = gauss_point[0]*j
            # Se evalua el PDE (Elasticidad, Timoshenko, Bernoulli, etc)
            f_int = eval(ProblemData.pde +'(f_int, x_element, N, dN, ddN, ProblemData, ElementData, ModelData, dX, "VectorF")')
            f_e = f_e + f_int
        # Se mapea los grados de libertad
        dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
        # Ensamblando
        # print(dof,f_e)
        f[dof] = f[dof] + f_e
        f_e = 0.0
    return f

def ApplyBC(A, f, BC_data, Mesh, ElementData,ProblemData, ModelData):
    '''Esta función aplica las condiciones de borde especificadas
    Input:
            A:      Matriz del sistema global antes de aplicar las CB
            f:      Vector de fuerzas del sistema global antes de aplicar las CB
            Mesh:   Clase que contiene los nodos y conexiones
                    obtenidos del mallado.
            Data:   Información del EF, problema a resolver y del
                    modelo utilizado.
    Output:
            A:      Matriz del sistema global después de aplicar las CB
            f:      Vector de fuerzas del sistema global después de aplicar las CB.       
    '''
    for bc in BC_data:
        if int(bc[1]) == 0:  # Neumann
            dof = int(ElementData.dof*bc[0]+bc[2])-1
            print("CB Neumann, DOF:",dof)
            if A[dof,dof] == 1.0: continue # ALREADY is Dirichlet BC
            f[dof] = f[dof] + bc[3]
        elif int(bc[1]) == 1:# Dirichlet
            dof = int(ElementData.dof*bc[0]+bc[2])-1
            print("CB Dirichlet, DOF:",dof)
            A[dof,:]  = 0.0
            A[dof,dof] = 1.0
            f[dof] = bc[3]
        else:
            print('Condición de Borde Desconocida')
    return A,f

def Elasticidad(A, X, N, dN,dNN, ProblemData, ElementData, ModelData, dX, tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    Input:
            - X:    Arreglo que contiene las coordenadas del EF.
            - N:    Arreglo que contiene las funciones de forma.
            - dN:   Arreglo que contiene las derivadas parciales de las
                    funciones de forma.
            - ddN:  Arreglo que contiene las segundas derivadas parciales
                    de las funciones de forma.
            - dX:   
            - tipo: Texto que indica que arreglo se quiere obtener. 
                    Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada
    Output:
            - A:    Arreglo del elemento finito que se quiere obtener.
    '''
    n, m, t = ElementData.nodes, ElementData.dof, ModelData.thickness
    #
    if tipo == 'MatrizK':
        E,v = ModelData.E, ModelData.v # Estado plano de esfuerzos
        E,v = E/(1.0-v*2),v/(1-v)# Estado plano de deformaciones
        if ProblemData.SpaceDim == 2:
            # Formando Matriz D
            D = zeros((3,3))
            D[0,0], D[1,1], D[0,1], D[1,0]= 1.0, 1.0, v, v
            D[2,2] = 0.5*(1.0-v)
            D = E*D/(1-v**2)
            # print("D=",D)
            # Formando Matriz B
            B = zeros((3,m*n))
            for i in range(m):
                B[i, i::m] = dN[i]
            B[2, 0::m] = dN[1]
            B[2, 1::m] = dN[0]
        else:
            print("Debes programar para %sD aún"%ProblemData.SpaceDim)
        #
        A = B.T@D@B*dX*t
    #
    elif tipo == 'MasaConsistente':
        Nmat = zeros((m, m*n))
        rho = ModelData.density
        for i in range(m):
            Nmat[i, i::m] = N[:]
        #
        A = rho*Nmat.T@Nmat*dX*t
    #
    elif tipo == 'MasaConcentrada':
        Nmat = zeros((m, m*n))
        rho = ModelData.density
        for i in range(m):
            Nmat[i, i::m] = N[:]
        #
        B = rho*Nmat.T@Nmat*dX*t
        one = zeros(m*n) + 1.0
        B = B@one
        A = zeros((m*n,m*n))
        # Concentrando Masas
        for i in range(m*n):
            A[i,i] = B[i]
    #
    elif tipo == 'VectorF':
        # Formando Matriz F
        Nmat=zeros((m,m*n))
        for i in range(m):
            Nmat[i, i::m] = N[:].T
            f = ModelData.selfweight*ModelData.gravity[0:m]
            A = Nmat.T@f*dX*t
    #
    else:
        print("Debes programar para el tipo %s aún"%tipo)
    return A


def CuadraturaGauss(puntos, dim):
    '''
    Esta función define los puntos de integración según la
    cuadratura de Gauss
    ---------------
    Input:
            puntos:     El número de puntos de integración de Gauss
            dim:        Dimensión del elemento finito
    Output:
            gp:         Arreglo cuya primera fila son los pesos y las
                        demás las posiciones respectivas                   
    '''
    if dim == 1:
        if puntos == 1:# Integración de Gauss de 1 punto en 1D
            gp     = zeros((2,1))
            gp[0], gp[1] = 2.0,0.0
            #
        elif puntos == 2:	# Integración de Gauss de 2 puntos en 1D
            gp     = zeros((2,2))
            gp[0,:] = 1.0
            gp[1,0], gp[1,1] =  -1/3**0.5, 1/3**0.5
            #
        elif puntos == 4: # Integración de Gauss de 4 puntos en 1D
            gp     = zeros((2,4))
            a, b = 0.33998104358484, 0.8611363115941
            wa,wb= 0.65214515486256, 0.34785484513744
            gp[0,0], gp[0,1], gp[0,2], gp[0,3] = wb,wa,wa,wb
            gp[1,0], gp[1,1], gp[1,2], gp[1,3] = -b,-a, a, b
        else:
            print("Debes programar para %s puntos aún."%puntos)
    elif dim == 2:
        if puntos == 1:## Integración de Gauss de 1 punto para Tri3
            gp=zeros((3,1))
            gp[0]=1.0*0.5 ##      peso*porcentaje del Jacobiano
            gp[1], gp[2]=1.0/3.0,1.0/3.0 ## Coordenadas de Integración
            #
        elif puntos == 4:## Integración de Gauss de 4 puntos para Quad4
            gp     = zeros((3,4))
            gp[0,:] = 1.0
            gp[1,0], gp[1,1], gp[1,2], gp[1,3] = -1.0/3**0.5, 1.0/3**0.5, 1.0/3**0.5,-1.0/3**0.5
            gp[2,0], gp[2,1], gp[2,2], gp[2,3] = -1.0/3**0.5,-1.0/3**0.5, 1.0/3**0.5, 1.0/3**0.5
        else:
            print("Debes programar para %s puntos aún"%puntos)
    else:
        print("Debes programar para %sD aún"%dim)
    
    return gp.T

def FunciónForma(X,gp,tipo):
    '''
    Esta función define funciones de forma y sus derivadas
    en coordenadas naturales para un elemento finito de
    coordenadas X, N(X)=N(xi),dN(X)=dN(xi)*J-1
    -----------------
    Input:
        X:      Matriz de coordenadas del elemento
        gp:     Arreglo que contiene los parametros de la cuadratura de Gauss
        tipo:   Tipo de elemento finito
    Output:
        N:      Matriz de funciones de Forma
        dN:     Matriz de la derivada de las funciones de Forma
        ddN:    Matriz de la segunda derivada de las funciones de Forma
        j:      Jacobiano para realizar la integración usando el mapeo isoparamétrico
    '''
    if tipo == 'Bar1':
        N, dN, J= zeros((1,2)), zeros((1,2)), zeros((1,1))
        #  
        ξ  = gp[1]
        N[0,0], N[0,1] = -ξ/2 + 0.5, ξ/2 + 0.5
        dN[0,0], dN[0,1]= -1/2,1/2
        #
    elif tipo == 'BarB':
        N, dN, ddN = zeros((1,4)), zeros((1,4)), zeros((1,4))
        #  
        ξ  = gp[1]
        le = X[1] - X[0]
        N[0,0] = le*(1 - ξ)**2*(1 + ξ)/8
        N[0,1] = (1 - ξ)**2*(2 + ξ)/4
        N[0,2] = le*(1 + ξ)**2*(-1 + ξ)/8
        N[0,3] = (1 + ξ)**2*(2 - ξ)/4
        dN[0,0] = -(1 - ξ)*(3*ξ + 1)/4
        dN[0,1] = -3*(1 - ξ)*(1 + ξ)/(2*le)
        dN[0,2] = (1 + ξ)*(3*ξ - 1)/4
        dN[0,3] = 3*(1 - ξ)*(1 + ξ)/(2*le)
        ddN[0,0] = (3*ξ - 1)/le
        ddN[0,1] = 6*ξ/(le**2)
        ddN[0,2] = (3*ξ + 1)/le 
        ddN[0,3] = -6*ξ/(le**2)
        j = le/2
        #
    elif tipo == 'Tri3':
        N, dN, J = zeros((3,1)), zeros((2,3)), zeros((2,2))
        #
        xi, eta = gp[1], gp[2]
        N[0],N[1],N[2] = xi,eta,1.0-xi-eta
        dN[0,0], dN[0,1], dN[0,2] =  1.0,  0.0, -1.0 #dN/d(xi)
        dN[1,0], dN[1,1], dN[1,2] =  0.0,  1.0, -1.0 #dN/d(eta)

    elif tipo == 'Quad4':
        N, dN, J= zeros((1,4)), zeros((2,4)), zeros((2,2))	
        a=array([-1.0, 1.0, 1.0,-1.0])# coordenadas x de los nodos 
        b=array([-1.0,-1.0, 1.0, 1.0])# coordenadas y de los nodos 	
        ξ = gp[1]
        η = gp[2]
        #
        N   = 0.25*(1.0 + a[:]*ξ)*(1.0 + b[:]*η) # 0.25(1+ξiξ)(1+ηiη)
        dN[0] = 0.25*a[:]*(1 + b[:]*η) # dN,ξ = 0.25ξi(1+ηiη)
        dN[1] = 0.25*b[:]*(1 + a[:]*ξ) # dN,η = 0.25ηi(1+ξiξ)
        #
    else:
        print("Debes programar para el tipo %s aún"%tipo)
    #
    # Calculamos la matriz jacobiana y su determinante
    try:
        j
    except NameError:
        J=dN@X
        if len(J) >1: 
            j = det(J)
            dN = inv(J)@dN
        else:
            j=J[0]
            dN = dN/j
    if j<0: 
        print("Cuidado: El jacobiano es negativo!")
        # print(X,'\n',dN,'\n',J,'\n',j)
    if tipo != 'BarB':
        ddN=0.0 # Retorna una matriz de ceros cuando no es Bernoulli
    #
    return N,dN,ddN,j

def DofMap(DofNode, connect, NodesElement):
    '''Función que mapea los grados de libertad correspondientes a un EF
    Input:
            DofNode:        Cantidad de grados de libertad por nodo
            connect:        Nodos del Elemento Finito
            NodesElement:   Cantidad de nodos del Elemento Finito
    Output:
            dof:            Lista que contiene los grados de libertad
                            del EF en el sistema global.
    '''
    dof=[]
    for i in range(NodesElement):
        for j in range(DofNode):
            dof.append( DofNode*connect[i] + j)
    return dof




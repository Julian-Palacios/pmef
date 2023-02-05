from numpy import zeros, array
from numpy.linalg import det, inv
from scipy.sparse import lil_matrix, csr_matrix
from time import time

# Funciones principales
def AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, MatrixType,showTime=False):
    '''Función que realiza el ensamble de la matriz A, del sistema Au = f.

    Input:
            Mesh:       Clase que contiene los nodos y conexiones
                        obtenidos del Mesh.
            __Data:     Información del EF, problema a resolver y del
                        modelo utilizado.
            MatrixType: Texto que indica que arreglo se quiere obtener
    Output:
            A:          Arreglo obtenido del ensamble de los arreglos A_e
                        de los elementos finitos.
    '''
    if showTime:
        print("\nEnsamble de K en el Sistema global...")
        start = time() 
    Ndof = Mesh.NN*ElementData.dof
    n = ElementData.nodes*ElementData.dof
    # Create ic, ir, data for sparse Matrix
    ir = zeros(Mesh.NC*n*n,'int')
    ic = zeros(Mesh.NC*n*n,'int')
    data = zeros(Mesh.NC*n*n,'float')
    cont = 0
    # Define Matriz para elementos
    A_e   = zeros((n,n), dtype='float64')
    A_int = zeros((n,n), dtype='float64')
    # Obtiene pesos y posiciones de la Cuadratura de Gauss
    gp = GaussianQuadrature(ElementData.noInt, ProblemData.SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento
    for connect_element in Mesh.Conex:
        # Obtiene coordenadas de nodos del elemento
        x_element = Mesh.Nodos[connect_element]
        # Bucle para realizar la integración según Cuadratura de Gauss
        for gauss_point in gp:
            # Evalua los puntos de Gauss en las Funciones de Forma
            [N, dN,ddN, j] = ShapeFunction(x_element, gauss_point, ElementData.type)
            dX = gauss_point[0]*j
            # Obtiene la Matriz de cada Elemento según el Problema(Elasticidad, Timoshenko, Bernoulli, etc)
            A_int = eval(ProblemData.pde +'(A_int, x_element, N, dN,ddN, ProblemData,ElementData, ModelData, dX, MatrixType)')
            A_e = A_e + A_int
        # Mapea los grados de libertad
        dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
        # Ensambla la matriz del elemento en la matriz global
        for i in range(n):
            for j in range(n):
                ir[cont] = dof[i]
                ic[cont] = dof[j]
                data[cont] = A_e[i,j]
                cont += 1
        # for k in dof: # loop for lil_matrix
        #     A[k,dof]=A[k,dof]+A_e[cont]
        #     cont += 1
        # Asigna 0 a la Matriz del Elemento
        A_e[:] = 0.0
    # Define matriz sparse para la matriz A global 
    A = csr_matrix((data,(ir,ic)),shape=(Ndof,Ndof),dtype='float')
    if showTime: print("AssembleMatrix demoró %.4f segundos"%(time()-start))
    return A
    
def AssembleVector(Mesh, ElementData, ProblemData, ModelData, MatrixType,showTime=False):
    '''Función que realiza el ensamble del vector f, del sistema Au = f.
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
    if showTime:
        print("\nEnsamble de f en el Sistema global...")
        start = time()
    N = Mesh.NN*ElementData.dof
    n = ElementData.nodes*ElementData.dof
    # Define matriz sparse para el vecto f global 
    f = zeros(N,'float64')
    # DefinE Matriz para elementos
    f_e    = zeros(n,'float64')
    f_int  = zeros(n,'float64')
    # Obtiene pesos y posiciones de la Cuadratura de Gauss
    gp = GaussianQuadrature(ElementData.noInt,ProblemData.SpaceDim)
    # Bucle para ensamblar la matriz de cada elemento si peso propio no es 0
    if ModelData.selfweight != 0.0:
        for connect_element in Mesh.Conex:
            # Obtiene coordenadas de nodos del elemento
            x_element = Mesh.Nodos[connect_element]
            # Bucle para realizar la integración según Cuadratura de Gauss
            for gauss_point in gp:
                # Evalua los puntos de Gauss en las Funciones de Forma
                [N, dN,ddN, j] = ShapeFunction(x_element, gauss_point,ElementData.type)
                dX = gauss_point[0]*j
                # Obtiene la Matriz de cada Elemento según el Problema(Elasticidad, Timoshenko, Bernoulli, etc)
                f_int = eval(ProblemData.pde +'(f_int, x_element, N, dN, ddN, ProblemData, ElementData, ModelData, dX, "VectorF")')
                f_e = f_e + f_int
            # Mapea los grados de libertad
            dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
            # Ensambla la matriz del elemento en la matriz global
            f[dof] = f[dof] + f_e
            # Asigna 0 a la Matriz del Elemento
            f_e = 0.0
    if showTime: print("AssembleVector demoró %.4f segundos"%(time()-start))
    return f

def ApplyBC(A, f, BC_data, Mesh, ElementData,ProblemData, ModelData,showBC=False,showTime=False):
    '''Esta función aplica las condiciones de borde especificadas.

    Input:
            A:      Matriz del sistema global antes de aplicar las CB.
            f:      Vector de fuerzas del sistema global antes de aplicar las CB.
            Mesh:   Clase que contiene los nodos y conexiones \
                    obtenidos del Mesh.
            Data:   Información del EF, problema a resolver y del \
                    modelo utilizado.
    Output:
            A:      Matriz del sistema global después de aplicar las CB.
            f:      Vector de fuerzas del sistema global después de aplicar las CB.       
    '''
    if showTime:
        print("\nAplicando Condiciones de Borde...")
        start = time()
    A = A.tolil()
    for bc in BC_data:
        if int(bc[1]) == 0:  # Neumann
            dof = int(ElementData.dof*bc[0] + bc[2]-1)
            if showBC==True: print("CB Neumann, DOF:",dof)
            if A[dof,dof] == 1.0: continue # ALREADY is Dirichlet BC
            f[dof] = f[dof] + bc[3]
        elif int(bc[1]) == 1:# Dirichlet
            dof = int(ElementData.dof*bc[0] + bc[2]-1)
            if showBC==True: print("CB Dirichlet, DOF:",dof)
            A[dof,:]  = 0.0
            A[dof,dof] = 1.0
            f[dof] = bc[3]
        else:
            print('Condición de Borde Desconocida')
    A = A.tocsr()
    if showTime: print("ApplyBC demoró %.4f segundos"%(time()-start))
    return A,f

def DofMap(DofNode, connect, NodesElement):
    '''Función que mapea los grados de libertad correspondientes a un EF.

    Input:
            DofNode:        Cantidad de grados de libertad por nodo
            connect:        Nodos del Elemento Finito
            NodesElement:   Cantidad de nodos del Elemento Finito
    Output:
            dof:            Lista que contiene los grados de libertad \
                            del EF en el sistema global.
    '''
    dof=zeros(NodesElement*DofNode,'int')
    cont = 0
    for i in range(NodesElement):
        for j in range(DofNode):
            dof[cont] = DofNode*connect[i] + j
            cont += 1
    return dof

# Funciones MEF
def Elasticity(A, X, N, dN,dNN, ProblemData, ElementData, ModelData, dX, tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss.

    Input:
        - X:    Arreglo que contiene las coordenadas del EF.
        - N:    Arreglo que contiene las funciones de forma.
        - dN:   Arreglo que contiene las derivadas parciales de las \
                funciones de forma.
        - ddN:  Arreglo que contiene las segundas derivadas parciales \
                de las funciones de forma. 
        - dX:   
        - tipo: Texto que indica que arreglo se quiere obtener. \
                Ejem: MatrizK, VectorF, MasaConsistente, MasaConcentrada
                
    Output:
        - A:    Arreglo del elemento finito que se quiere obtener.
    '''
    n, m = ElementData.nodes, ElementData.dof
    #
    if ProblemData.SpaceDim == 1:
        area = ModelData.area
        if tipo == 'MatrizK':
            E = ModelData.E
            # Formando Matriz B
            B = zeros((1,m*n))
            for i in range(m):
                B[i, i::m] = dN[i]
            A = B.T@B*dX*E*area

        elif tipo == 'VectorF':
            # Formando Matriz F
            Nmat=zeros((m,m*n))
            for i in range(m):
                Nmat[i, i::m] = N
                f = ModelData.selfweight*ModelData.gravity
                A = Nmat*f*area*dX

        else:
            print("Debes programar para el tipo %s aún."%tipo)

    elif ProblemData.SpaceDim == 2:
        t = ModelData.thickness

        if tipo == 'MatrizK':
            E,v = ModelData.E, ModelData.v # Estado plano de esfuerzos
            if ModelData.state == 'PlaneStress': pass
            elif ModelData.state == 'PlaneStrain': E,v = E/(1.0-v*2),v/(1-v)
            else: print('El estado plano solo puede ser "PlaneStress" o "PlaneStrain"')
            
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
            #
            A = B.T@D@B*dX*t
        #
        elif tipo == 'MasaConsistente':
            Nmat = zeros((m, m*n))
            rho = ModelData.density
            for i in range(m):
                Nmat[i, i::m] = N
            #
            A = rho*Nmat.T@Nmat*dX*t
        #
        elif tipo == 'MasaConcentrada':
            Nmat = zeros((m, m*n))
            rho = ModelData.density
            for i in range(m):
                Nmat[i, i::m] = N
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
                Nmat[i, i::m] = N
                f = ModelData.selfweight*ModelData.gravity[0:m]
                A = Nmat.T@f*dX*t
        #
        else:
            print("Debes programar para el tipo %s aún"%tipo)

    elif ProblemData.SpaceDim == 3:
        if tipo == 'MatrizK':
            E, v = ModelData.E, ModelData.v
            # Formando Matriz D
            D = zeros((6, 6))
            λ = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
            μ = E / (2.0 * (1.0 + v))

            D[0, 0] = D[1, 1] = D[2, 2] = λ + 2*μ 
            D[3, 3] = D[4, 4] = D[5, 5] = μ
            D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = λ

            # Formando Matriz B
            B = zeros((6, m * n))

            for i in range(m):
                B[i, i::m] = dN[i]

            B[3, 0::m] = dN[1]
            B[3, 1::m] = dN[0]
            B[4, 1::m] = dN[2]
            B[4, 2::m] = dN[1]
            B[5, 0::m] = dN[2]
            B[5, 2::m] = dN[0]

            A = B.T@D@B *dX
        #
        elif tipo == 'MasaConsistente':
            Nmat = zeros((m, m*n))
            rho = ModelData.density
            for i in range(m):
                Nmat[i, i::m] = N
            #
            A = rho*Nmat.T@Nmat*dX# *t
        #
        elif tipo == 'MasaConcentrada':
            Nmat = zeros((m, m*n))
            rho = ModelData.density
            for i in range(m):
                Nmat[i, i::m] = N
            #
            B = rho*Nmat.T@Nmat*dX# *t
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
                Nmat[i, i::m] = N
                f = ModelData.selfweight*ModelData.gravity[0:m]
                A = Nmat.T@f*dX # *t
                
        else:
            print("Debes programar para el tipo %s aún"%tipo)
    else:
        print("Debes programar para %sD aún"%ProblemData.SpaceDim)
    return A

def GaussianQuadrature(puntos, dim):
    '''
    Esta función define los puntos de integración según la \
    cuadratura de Gauss
    
    Input:
            puntos:     El número de puntos de integración de Gauss
            dim:        Dimensión del elemento finito
    Output:
            gp:         Arreglo cuya primera fila son los pesos y las \
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
            a = 0.5773502691896257
            gp[1,0], gp[1,1] =  -a, a
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
            a = 0.5773502691896257
            gp[1,0], gp[1,1], gp[1,2], gp[1,3] = -a, a, a,-a
            gp[2,0], gp[2,1], gp[2,2], gp[2,3] = -a,-a, a, a
        else:
            print("Debes programar para %s puntos aún"%puntos)
    elif dim == 3:

        if puntos == 8:
            a = 3**-0.5#0.5773502691896257

            gp = zeros((4, 8))
            gp[0, :] = 1.0
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1, 3], gp[1, 4], gp[1, 5], gp[
                1, 6], gp[1, 7] = -a, a, a, -a, -a, a, a, -a
            gp[2, 0], gp[2, 1], gp[2, 2], gp[2, 3], gp[2, 4], gp[2, 5], gp[
                2, 6], gp[2, 7] = -a, -a, a, a, -a, -a, a, a
            gp[3, 0], gp[3, 1], gp[3, 2], gp[3, 3], gp[3, 4], gp[3, 5], gp[
                3, 6], gp[3, 7] = -a, -a, -a, -a, a, a, a, a
        else:
            print("Debes programar para %s puntos aún." %puntos)

    else:
        print("Debes programar para %sD aún."%dim)
    
    return gp.T

def ShapeFunction(X,gp,tipo):
    '''
    Esta función define funciones de forma y sus derivadas \
    en coordenadas naturales para un elemento finito de \
    coordenadas X, N(X)=N(xi),dN(X)=dN(xi)*J-1
    
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
        N, dN, J = zeros((1,3)), zeros((2,3)), zeros((2,2))
        #
        xi, eta = gp[1], gp[2]
        N[0,0],N[0,1],N[0,2] = xi,eta,1.0-xi-eta
        dN[0,0], dN[0,1], dN[0,2] =  1.0,  0.0, -1.0 # dN,ξ 
        dN[1,0], dN[1,1], dN[1,2] =  0.0,  1.0, -1.0 # dN,η

    elif tipo == 'Quad4':
        N, dN, J= zeros((1,4)), zeros((2,4)), zeros((2,2))	
        a=array([-1.0, 1.0, 1.0,-1.0])# coordenadas x de los nodos 
        b=array([-1.0,-1.0, 1.0, 1.0])# coordenadas y de los nodos 	
        ξ = gp[1]
        η = gp[2]
        #
        N   = 0.25*(1.0 + a[:]*ξ)*(1.0 + b[:]*η) # N = 0.25(1+ξiξ)(1+ηiη)
        dN[0] = 0.25*a[:]*(1 + b[:]*η) # dN,ξ = 0.25ξi(1+ηiη)
        dN[1] = 0.25*b[:]*(1 + a[:]*ξ) # dN,η = 0.25ηi(1+ξiξ)
        #
    elif tipo == 'Brick8':

        N, dN, J = zeros((1, 8)), zeros((3, 8)), zeros((3, 3))
        a = array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
        b = array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        c = array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        ξ = gp[1]
        η = gp[2]
        ζ = gp[3]

        N = 0.125 * (1.0 + a[:] * ξ) * (1.0 + b[:] * η) * (1.0 + c[:] * ζ)

        dN[0] = 0.125 * a[:] * (1 + b[:] * η) * (1.0 + c[:] * ζ)
        dN[1] = 0.125 * b[:] * (1 + a[:] * ξ) * (1.0 + c[:] * ζ)
        dN[2] = 0.125 * c[:] * (1 + a[:] * ξ) * (1.0 + b[:] * η)

    else:
        print("Debes programar para el tipo %s aún"%tipo)
    
    # Calculamos la matriz jacobiana y su determinante
    try: j
    except NameError:
        J=dN@X
        if len(J) >1: 
            j = abs(det(J))
            dN = inv(J)@dN
        else:
            j = abs(J[0])
            dN = dN/j
    if j<0: 
        print("Cuidado: El jacobiano es negativo!")
        # print(X,'\n',dN,'\n',J,'\n',j)
    if tipo != 'BarB':
        ddN=0.0 # Retorna una matriz de ceros cuando no es Bernoulli
    #
    return N,dN,ddN,j


def Bernoulli(A, x, N, dN, ddN, ProblemData, ElementData, ModelData, dX, tipo):
    '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
    '''
    n = ElementData.nodes
    m = ElementData.dof
    EI = ModelData.EI
    Area = ModelData.Area
    ##
    if tipo == 'MatrizK':
        A = EI*ddN.T@ddN
        A = A*dX
        # print(ddN,A)
    elif tipo == 'MasaConsistente':
        rho = ModelData.density
        A = rho*N.T@N*dX*Area
        ##
    elif tipo == 'MasaConcentrada':
        rho = ModelData.density
        A = zeros((m*n, m*n))
        for i in range(m*n):
            if i%2==1: A[i, i] = 0.5*rho*dX*Area
            else: A[i, i] = 0.005*rho*dX*Area
        ##
    elif(tipo == 'VectorF'):
        # Formando matriz N para v
        N_v = zeros((m*n, 1))
        N_v = N
        A = zeros((m*n, 1))
        f = ModelData.fy
        A = N_v*f*dX
        ##
    else:
        print("Debes programar para el tipo %s aún" % tipo)
    return A


def Timoshenko(A, x, N, dN, ddN, ProblemData, ElementData, ModelData, dX, tipo):
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
            N_theta = zeros((1, m*n))
            N_theta[0, 0::m] = N[0]
            # Formando N para v
            N_v = zeros((1, m*n))
            N_v[0, 1::m] = N[0]
            # Formando B para theta
            B_theta = zeros((1, m*n))
            B_theta[0, 0::m] = dN[0]
            # Formando B para for v
            B_v = zeros((1, m*n))
            B_v[0, 1::m] = dN[0]
            ##
        elif (ProblemData.SpaceDim == 2):
            print('Solo es válido para 1D')
        elif (ProblemData.SpaceDim == 3):
            print('Solo es válido para 1D')
        ##
        # Calculando Matriz
        A = B_theta.T*EI@B_theta + N_theta.T*GAs@N_theta - N_theta.T*GAs@B_v
        A = A - B_v.T*GAs@N_theta + B_v.T*GAs@B_v
        A = A*dX
    ##
    elif tipo == 'MasaConsistente':
        rho = ModelData.density
        Nmat = zeros((1,4),'float')
        Nmat[:,1::2] = N
        A = rho*Nmat.T@Nmat*dX*Area
        # Artificio para no obetener una Matriz singular
        A[0, 0] = A[0, 0]+0.01*A[1, 1]
        A[2, 2] = A[2, 2]+0.01*A[3, 3]
    ##
    elif tipo == 'MasaConcentrada':
        rho = ModelData.density
        Nmat = zeros((1,4),'float')
        Nmat[:,1::2] = N
        B = rho*Nmat.T@Nmat*dX*Area
        # Artificio para no obetener una Matriz singular
        B[0, 0] = B[0, 0]+0.01*B[1, 1]
        B[2, 2] = B[2, 2]+0.01*B[3, 3]
        ##
        one = zeros(m*n)
        one[:] = 1.0
        B = B@one
        A = zeros((m*n, m*n))
        # Concentrando Masas
        for i in range(m*n):
            A[i, i] = B[i]

    elif(tipo == 'VectorF'):
        # Formando matriz N para theta
        N_theta = zeros((1, m*n))
        N_theta[0, 0::m] = N[0]
        # Formando matriz N para v
        N_v = zeros((1, m*n))
        N_v[0, 1::m] = N[0]
        #
        f = ModelData.fy
        A = N_v*f*dX

    else:
        print("Debes programar para el tipo %s aún" % tipo)
    return A

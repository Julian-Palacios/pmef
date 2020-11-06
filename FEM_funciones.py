## Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe) 
## Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
## Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##
## Este código se basó en las rutinas de matlab FEMCode
## realizado inicialmente por Garth N .Wells (2005)
## para la clase CT5123 en TU Delft, Países Bajos
## 

import numpy as np
from numpy.linalg import det, inv
from scipy import sparse

def GenQuadMesh(L,H,lc,fd="./INPUT"):
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
  gmsh.write(fd+"/rectangle_%4.2f.msh"%lc)
##  gmsh.fltk.run() ##Abre el gmsh
  gmsh.finalize()
  return fd+"/rectangle_%4.2f.msh"%lc

def gmsh_read(msh_file, ProblemData, ElementData):
  '''Función que retorna los nodos y conexiones al leer el archivo .msh
    realizado por gmsh (función adaptada del código en matlab FEMCode).
  '''
  print('Leyendo archivo msh')
  file_input   = open(msh_file,'rt')
# ------------- Test input file ----------------------
# Read 2 first lines and check if we have mesh format 2
  mesh_format = file_input.readline()[:-1]
  line = file_input.readline()
  fmt = int(float(line.split()[0]))
  if( mesh_format != '$MeshFormat' or fmt != 2):
      print('The mesh format is NOT version 2!')

# -------------------- Nodes  --------------------------
# Process file until we get the line with the number of nodes
  buf = file_input.readline()[:-1]
  while('$Nodes' != buf):
    buf = file_input.readline()[:-1]
  noNodes = int(file_input.readline()[:-1])# Extract number of nodes
# Initialise nodes matrix [x1, y1, z1 x2, y2, z2 .... xn, yn, zn]
  x = np.zeros((noNodes,3),dtype=np.float64)
  for i in range(noNodes):# Get nodal coordinates
    buf=[float(y) for y in file_input.readline()[:-1].split()]
    x[i] = buf[1:4]#we throw away the node numbers!

# ------------ Elements --------------------
# Process file until we get the line with the number of elements
  while('$Elements' != buf):
    buf = file_input.readline()[:-1]
# Extract number of elements
  noElements = int(file_input.readline()[:-1])
# Get first line of connectivity
  buf = [int(y) for y in file_input.readline()[:-1].split()]
# Number of nodes per element
  no_nodes_per_elem = len(buf) - (3 + buf[2])
  tipo = buf[1]# Get type of element
# Verify that we have the correct element
  if(no_nodes_per_elem != ElementData.nodes):# Check number of nodes
    print('The number of nodes per element in the mesh differ from ElementData.nodes')
# Check element type (gmsh 2.0 manual )
  if(ElementData.type == 'Tri3'):
    if(tipo != 2):
      print('Element type is not Tri3')
  elif(ElementData.type == 'Quad4'):
    if(tipo !=3 ):
      print('Element type is not Quad4')
  elif(ElementData.type == 'Tri6'):
    if(tipo != 9):
      print('Element type is not Tri6')
  elif(ElementData.type == 'Tet4'):
    if(tipo !=4 ):
      print('Element type is not Tet4')
  else: # Default error message
    print('Element type %s is not supported', ElementData.type)

# --------- Initialise connecticity matrix and write first line ------------
  connect = np.zeros((noElements, no_nodes_per_elem),dtype=np.int32)
  connect[0,:] = buf[3 + buf[2]:len(buf)]
# Get element connectivity
# FIXME: check that the nodes on the elements are numbered correctly!
  for i in range(1,noElements):
    buf = [int(y) for y in file_input.readline()[:-1].split()]
    # Only one type of elements is allowed in the mesh
    if(tipo != buf[1]):
      print('More than one type of elements is present in the mesh, did you save all elements?')
    connect[i,:] = buf[3 + buf[2]:len(buf)]# throw away element number, type and arg list

# ------- Clean up and close -----------------------
# Delete coordinates
  if (ProblemData.SpaceDim == 1):
    x[:,1:3] = 0.0
  elif (ProblemData.SpaceDim == 2):
    x = np.delete(x, 2, 1)
  file_input.close()# Close file

# ------- Add members to object  ---------------
  class Mesh:
    NN = noNodes
    NC = noElements
    Nodos = x.T
    Conex = connect.T-1
  return Mesh

def LinearMesh(L,N,x0=0):
  '''Función que retorna los nodos y conexiones de un Mesh especificado
     para un elemento finito (EF) en 1D.
  '''
  L_element=L/N
  nodos=np.zeros(N+1)
  for i in range(N+1): nodos[i]=x0 + i*L_element
  conex=np.zeros((N,2),dtype=np.uint8)
  conex[:,0],conex[:,1]=range(N),range(1,N+1)
  return nodos,conex

def genBC_2D(BC_coord,X,lim=0.01):
  ''' Función que aplica las condiciones de borde en un punto especificando
      las coordenadas donde se encuentra. (Utiliza un radio de búsqueda)
  '''
  NN,k=len(X),0
  BC_data=np.zeros((len(BC_coord),4))
  BC_data[:,1:]=BC_coord[:,2:]
  for x,y in BC_coord[:,0:2]:
    for i in range(NN):
      if i==0:
        er0,ind=((X[i,0]-x)**2+(X[i,1]-y)**2)**0.5,0
        continue
      er=((X[i,0]-x)**2+(X[i,1]-y)**2)**0.5
      if er<er0:
        er0,ind=er,i
        continue
      ##
    if er0<lim:
      BC_data[k,0]=ind+1
      k=k+1
    else:
      print("No se encuentra un punto cerca a (%s,%s) en un radio de %s"%(x,y,lim))
      k=k+1
  return BC_data

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
    if puntos == 1:# Integración de Gauss de 1 punto en 1D
      gp     = np.zeros((2,1))
      gp[0], gp[1] = 2.0,0.0
	
    elif puntos == 2:	# Integración de Gauss de 2 puntos en 1D
      gp     = np.zeros((2,2))
      gp[0,0],gp[1,0] =  1.0,-1/3**0.5
      gp[0,1],gp[1,1] =  1.0,1/3**0.5
      
    elif puntos == 4:
      gp     = np.zeros((2,4))
      a,b = 0.33998104358484, 0.8611363115941
      wa,wb = 0.65214515486256, 0.34785484513744
      gp[0,0],gp[1,0] =  wb,-b
      gp[0,1],gp[1,1] =  wa,-a
      gp[0,2],gp[1,2] =  wa,a
      gp[0,3],gp[1,3] =  wb,b
    else:
      print("Debes programarpar %s puntos aún"%puntos)
  ##
  elif dim == 2:
    if puntos == 1:## Integración de Gauss de 1 punto para Tri3
      gp=np.zeros((3,1))
      gp[0]=1.0*0.5 ##      peso*porcentaje del Jacobiano
      gp[1], gp[2]=1.0/3.0,1.0/3.0 ## Coordenadas de Integración
      #
    elif puntos == 4:	# Integración de Gauss de 2x2 en 2D para Quad
      gp     = np.zeros((3,4))
      gp[0,:] = 1.0
      gp[1,0], gp[1,1], gp[1,2], gp[1,3] = -1.0/3**0.5, 1.0/3**0.5, 1.0/3**0.5,-1.0/3**0.5
      gp[2,0], gp[2,1], gp[2,2], gp[2,3] = -1.0/3**0.5,-1.0/3**0.5, 1.0/3**0.5, 1.0/3**0.5
    else: print("Debes programarpar %s puntos aún"%puntos)
  ##
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
    N, dN, J= np.zeros((2,1)), np.zeros((2,1)), np.zeros((1,1))
    #  
    xi  = gp[1]
    N[0], N[1] = -xi/2 + 0.5, xi/2 + 0.5
    dN[0,0], dN[1,0]= -1/2,1/2
    #
  elif tipo == 'BarB':
    N, dN, ddN, J= np.zeros((4,1)), np.zeros((4,1)), np.zeros((4,1)), np.zeros((1,1))
    #  
    xi  = gp[1]
    le = X[1] - X[0]
    N[0] = le*(1 - xi)**2*(1 + xi)/8
    N[1] = (1 - xi)**2*(2 + xi)/4
    N[2] = le*(1 + xi)**2*(-1 + xi)/8
    N[3] = (1 + xi)**2*(2 - xi)/4
    dN[0] = -(1 - xi)*(3*xi + 1)/4
    dN[1] = -3*(1 - xi)*(1 + xi)/(2*le)
    dN[2] = (1 + xi)*(3*xi - 1)/4
    dN[3] = 3*(1 - xi)*(1 + xi)/(2*le)
    ddN[0] = (3*xi - 1)/le
    ddN[1] = 6*xi/(le**2)
    ddN[2] = (3*xi + 1)/le 
    ddN[3] = -6*xi/(le**2)
    j = le/2
    return N,dN,ddN,j
    #
  elif tipo == 'Tri3':
    N, dN, J = np.zeros((3,1)), np.zeros((3,2)), np.zeros((2,2))
    #
    xi, eta = gp[1], gp[2]
    N[0],N[1],N[2] = xi,eta,1.0-xi-eta
    dN[0,0], dN[1,0], dN[2,0] =  1.0,  0.0, -1.0 #dN/d(xi)
    dN[0,1], dN[1,1], dN[2,1] =  0.0,  1.0, -1.0 #dN/d(eta)
    #
  elif tipo == 'Quad4':
    N, dN, J= np.zeros((4,1)), np.zeros((4,2)), np.zeros((2,2))	
    a=np.array([-1.0, 1.0, 1.0,-1.0])# coordenadas x de los nodos 
    b=np.array([-1.0,-1.0, 1.0, 1.0])# coordenadas y de los nodos 	
    xi  = gp[1]
    eta = gp[2]
    ##
    N   = 0.25*(1.0 + a[:]*xi + b[:]*eta + a[:]*b[:]*xi*eta)
    dN[:,0] = 0.25*(a[:] + a[:]*b[:]*eta)
    dN[:,1] = 0.25*(b[:] + a[:]*b[:]*xi)
    #
  else:
    print("Debes programar para el tipo %s aún"%tipo)
  ## Calculamos la matriz jacobiana y su determinante
  J=X@dN
  ##
  if len(J)>1: j=det(J); dN = dN@inv(J)
  else: j=J[0]; dN = dN/j
  if(j<0): print("Cuidado: El jacobiano es negativo!")
##  print(X,'\n',dN,'\n',J,'\n',j)
  ddN=0.0## Retorna 0.0 cuando no es Bernoulli
  return N,dN,ddN,j

###### Ejemplo de aplicacion de las funciones de forma para Tri3
####X=Mesh.Nodos[:,Mesh.Conex[:,0]]#np.array([[3,2],[8,7],[1,12]]).T
####gp=CuadraturaGauss(1,2)
####[N,dN,ddN,j]=FunciónForma(X,gp.T,tipo='Tri3')

def Bernoulli(A, x, N, dN, ddN, ProblemData, ElementData, ModelData, dX, tipo):
  '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
  '''
  n   = ElementData.nodes
  m   = ElementData.dof
  EI  = ModelData.EI
  Area = ModelData.Area
  ##
  if tipo=='MatrizK':
    A = EI*ddN@ddN.T
    A = A*dX
##    print(A)
  elif tipo == 'MasaConsistente':
    rho = ModelData.density
    A = rho*N@N.T*dX*Area
    ##
  elif tipo == 'MasaConcentrada':
    rho = ModelData.density
    B = rho*N@N.T*dX*Area
    ##
    one = np.zeros(m*n)
    one[:] = 1.0
    B = B@one
    A = np.zeros((m*n,m*n))
    # Concentrando Masas (Mejorar proceso si |A| <= 0)
    for i in range(m*n):
      A[i,i] = B[i]
    ##
  elif(tipo=='VectorF'):
    # Formando matriz N para v
    N_v = np.zeros((m*n,1))
    N_v = N
    A = np.zeros((m*n,1))
    f = ModelData.fy
    A = N_v.T*f*dX
    ##
  else:
    print("Debes programar para el tipo %s aún"%tipo)
  return A

def Timoshenko(A, x, N, dN,ddN, ProblemData, ElementData, ModelData,dX, tipo):
  '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
  '''
  n   = ElementData.nodes
  m   = ElementData.dof
  EI  = ModelData.EI
  GAs = ModelData.GAs
  Area=ModelData.Area
  ##
  if tipo=='MatrizK':
    if (ProblemData.SpaceDim == 1):
      # Formando N para theta
      N_theta = np.zeros((1, m*n))
      N_theta[0, 0::m] = N[:,0].T
      # Formando N para v
      N_v = np.zeros((1, m*n))
      N_v[0, 1::m] = N[:,0].T
      # Formando B para theta
      B_theta = np.zeros((1, m*n))
      B_theta[0, 0::m] = dN[:,0].T
      # Formando B para for v
      B_v = np.zeros((1, m*n))
      B_v[0, 1::m] = dN[:,0].T
      ##
    elif (ProblemData.SpaceDim == 2):
      print( 'Solo es válido para 1D' )
    elif (ProblemData.SpaceDim == 3):
      print( 'Solo es válido para 1D' )
    ##
    #Calculando Matriz
    A = B_theta.T*EI@B_theta + N_theta.T*GAs@N_theta - N_theta.T*GAs@B_v
    A = A - B_v.T*GAs@N_theta + B_v.T*GAs@B_v
    A = A*dX
  ##
  elif tipo == 'MasaConsistente':
    rho = ModelData.density
    nulo=np.array([0])
    N=np.array([nulo,N[0],nulo,N[1]])
    A = rho*N@N.T*dX*Area
    ## Artificio para no obetener una Matriz singular
    A[0,0]=A[0,0]+0.01*A[1,1]
    A[2,2]=A[2,2]+0.01*A[3,3]
  ##
  elif tipo == 'MasaConcentrada':
    rho = ModelData.density
    nulo=np.array([0])
    N=np.array([nulo,N[0],nulo,N[1]])
    B = rho*N@N.T*dX*Area
    ## Artificio para no obetener una Matriz singular
    B[0,0]=B[0,0]+0.01*B[1,1]
    B[2,2]=B[2,2]+0.01*B[3,3]
    ##
    one = np.zeros(m*n)
    one[:] = 1.0
    B = B@one
    A = np.zeros((m*n,m*n))
    # Concentrando Masas
    for i in range(m*n):
      A[i,i] = B[i]

  elif(tipo=='VectorF'):
    # Formando matriz N para theta
    N_theta = np.zeros((1, m*n))
    N_theta[0, 0::m] = N[:,0].T
    # Formando matriz N para v
    N_v = np.zeros((1, m*n))
    N_v[0, 1::m] = N[:,0].T
    #
    f = ModelData.fy
    A = N_v*f*dX
  
  else:
    print("Debes programar para el tipo %s aún"%tipo)
  return A

def Elasticidad(A, X, N, dN,dNN, ProblemData, ElementData,ModelData, dX, tipo):
  '''Función que retorna la matriz de un EF evaluado en un Punto de Gauss
  '''
  n = ElementData.nodes
  m = ElementData.dof
  t = ModelData.thickness
  if tipo == 'MatrizK':
    E,v = ModelData.E, ModelData.v## Estado plano de esfuerzos
##    E,v = E/(1.0-v*2),v/(1-v)## Estado plano de deformaciones
    if ProblemData.SpaceDim == 2:
      ## Formando Matriz D
      D = np.zeros((3,3))
      D[0,0], D[1,1]= 1.0, 1.0
      D[0,1], D[1,0] = v, v
      D[2,2] = 0.5*(1.0-v)
      D=E*D/(1-v**2)
##      print("D=",D)
      ## Formando Matriz B
      B = np.zeros((3,m*n))
      for i in range(m):
        B[i, i::m] = dN[:,i]
      B[2, 0::m] = dN[:,1]
      B[2, 1::m] = dN[:,0]
    else:
      print("Debes programar para %sD aún"%dim)
    #
    A = B.T@D@B*dX*t

  elif tipo == 'MasaConsistente':
    Nmat = np.zeros((m, m*n))
    rho = ModelData.density
    for i in range(m):
      Nmat[i, i::m] = N[:].T
    #####
    A = rho*Nmat.T@Nmat*dX*t
      
  elif tipo == 'MasaConcentrada':
    Nmat = np.zeros((m, m*n))
    rho = ModelData.density
    for i in range(m):
      Nmat[i, i::m] = N[:].T
    ####
    B = rho*Nmat.T@Nmat*dX*t
    one = np.zeros(m*n)
    one[:] = 1.0
    B = B@one
    A = np.zeros((m*n,m*n))
    # Concentrando Masas
    for i in range(m*n):
      A[i,i] = B[i]
    
  elif tipo == 'VectorF':
    ##Formando Matriz F
    Nmat=np.zeros((m,m*n))
    for i in range(m):
      Nmat[i, i::m] = N[:].T
    f = ModelData.selfweight*ModelData.gravity[0:m]
    ##
    A = Nmat.T@f*dX*t
  else:
    print("Debes programar para el tipo %s aún"%tipo)
  return A
  
def DofMap(DofNode, connect, NodesElement):
  '''Función que mapea los grados de libertad correspondientes a un EF
  '''
  dof=[]
  for i in range(NodesElement):
    for j in range(DofNode):
      dof.append( DofNode*(connect[i]) + j)
  return dof

def AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, MatrixType):
  '''Función que realiza el ensamble de la matriz K, de Ku=F
  '''
  N = Mesh.NN*ElementData.dof
  n = ElementData.nodes*ElementData.dof
  # Definiendo matriz sparse para la matriz global 
  A = sparse.lil_matrix((N,N),dtype=np.float64)
  # Definiendo Matriz para elementos
  A_e   = np.zeros(n,dtype=np.float64)
  A_int = np.zeros(n,dtype=np.float64)
  # Se obtiene los pesos y posiciones de la Cuadratura de Gauss
  gp = CuadraturaGauss(ElementData.noInt,
                         ProblemData.SpaceDim)
  # Bucle para ensamblar la matriz de cada elemento
  for element in Mesh.Conex.T:
    # Asigando coordenadas de nodos y connectividad
    connect_element = element[0:ElementData.nodes]
    if ProblemData.SpaceDim == 1: x_element = Mesh.Nodos[connect_element]
    else: x_element       = Mesh.Nodos[:,connect_element]
    # Bucle para realizar la integración según Cuadratura de Gauss
    for gauss_point in gp:
      # Se calcula Las Funciones de Forma
      [N, dN,ddN, j] = FunciónForma(x_element, gauss_point, ElementData.type)
      # Se calcula la Matriz de cada Elemento
      dX = gauss_point[0]*j
      ##Evaluar el PDE (Elasticidad, Timoshenko, Bernoulli, etc)
      A_int = eval(ProblemData.pde +'(A_int, x_element, N, dN,ddN, ProblemData,ElementData, ModelData, dX, MatrixType)')
      A_e = A_e + A_int
##    if MatrixType=="MatrizK": print("K_elemento",A_e)
    # Se mapea los grados de libertad
    dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
  # Ensamblando
    cont=0
    for k in dof:
      A[k,dof]=A[k,dof]+A_e[cont]
      cont=cont+1
    # Se resetea la Matriz del Elemento
    A_e[:] = 0.0
  return A

def AssembleVector(Mesh, ElementData, ProblemData, ModelData,
                   MatrixType):
  '''Función que realiza el ensamble del vector F, de Ku=F
  '''
  N = Mesh.NN*ElementData.dof
  n = ElementData.nodes*ElementData.dof
  # Definiendo vector f global 
  f = np.zeros(N,np.float64)
  # Definiendo Matriz para elementos
  f_e    = np.zeros(n,np.float64)
  f_int  = np.zeros(n,np.float64)
  # Se obtiene los pesos y posiciones de la Cuadratura de Gauss
  gp = CuadraturaGauss(ElementData.noInt,ProblemData.SpaceDim)
  # Bucle para ensamblar la matriz de cada elemento
  for element in Mesh.Conex.T:
    # Asigando coordenadas de nodos y connectividad
    connect_element = element[0:ElementData.nodes]
    if ProblemData.SpaceDim == 1: x_element = Mesh.Nodos[connect_element]
    else: x_element       = Mesh.Nodos[:,connect_element]
    # Bucle para realizar la integración según Cuadratura de Gauss
    for gauss_point in gp:
      # Se calcula Las Funciones de Forma
      [N, dN,ddN, j] = FunciónForma(x_element, gauss_point,ElementData.type)      
      # Se calcula la Matriz de cada Elemento
      dX = gauss_point[0]*j
      ##Evaluar el PDE (Elasticidad, Timoshenko, Bernoulli, etc)
      f_int = eval(ProblemData.pde +'(f_int, x_element, N, dN, ddN, ProblemData, ElementData, ModelData, dX, "VectorF")')
      f_e = f_e + f_int
    # Se mapea los grados de libertad
    dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
  # Ensamblando
##    print(dof,f_e)
    f[dof] = f[dof] + f_e
    f_e = 0.0
  return f

def ApplyBC(A, f, BC_data, Mesh, ElementData,ProblemData, ModelData):
  '''Esta función aplica las condiciones de borde especificadas
  '''
  for bc in BC_data:
    if int(bc[1]) == 0:  # Neumann
      dof = int(ElementData.dof*(bc[0]-1)+bc[2])-1
      print("Neumann, DOF:",dof)
      f[dof] = f[dof]+bc[3]
    elif int(bc[1]) == 1:#Dirichlet
      dof = int(ElementData.dof*(bc[0]-1)+bc[2])-1
      print("Dirichlet, DOF:",dof)
      A[dof,:]  = 0.0
      A[dof,dof] = 1.0
      f[dof] = bc[3]
    else:
     print('Condición de Borde Desconocida')
  return A,f

########## Funciones Diversas ############

def Deformada(X,u,FS=1/500):
  ''' Función que agrega una deformación a la pocisión de Nodos
  '''
  NN=len(X)
  X_def=np.zeros(X.shape)
  for i in range(NN):
    X_def[i]=X[i]+FS*np.array([u[2*i],u[2*i+1]])
  return X_def
  
def tridiag(a=2.1, n=5):
  ''' Función que retorna una matriz de rigidez uniforme de a para n gdl
  '''
  aa=[-a for i in range(n-1)]
  bb=[2*a for i in range(n)]
  bb[-1]=a
  cc=[-a for i in range(n-1)]
  return np.diag(aa, -1) + np.diag(bb, 0) + np.diag(cc, 1)

def K_reductor(K,dof):
  '''Función que elimina grados de libertad que no se desea analizar
  '''
  k=0
  for i in dof:
    K=np.delete(np.delete(K,i-k,0),i-k,1)
    k=k+1
##   print(i,len(K))
  return K

def V_insertor(V,dof):
  '''Función que agrega valores nulos aun vector en pocisiones especificadas
  '''
  for i in dof:
        V=np.insert(V,i,0)
  return V

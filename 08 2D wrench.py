from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, plot2D_deform, stress, graph, K_reduce, V_insert

import time
from numpy import array, zeros, append, all, pi
from scipy.sparse.linalg import spsolve, eigsh
from numpy.linalg import solve, det
import matplotlib.pyplot as plt

# Unidades
mm = 0.001
cm = 0.01
kgf = 9.80665
tonf = 1000*kgf

gravity = array([0.0,-1.0,0.0])
class ProblemData:
    SpaceDim = 2
    pde="Elasticity"
class ModelData:
    E  = 2e11
    v = 0.3
    thickness = 2*mm
    state = 'PlaneStress'
    density = 7860
    selfweight= 0.0
    gravity = gravity

import meshio
mesh = meshio.read('2Dwrench.msh')
cells = {"quad": mesh.cells_dict['quad']}

class ElementData:
    dof = 2
    nodes = 4
    noInt =  4
    type = 'Quad4'

x, cnx = mesh.points[:,:2]*mm, mesh.cells_dict['quad']
print('# Nodos: %i \n# Elementos: %i'%(len(x),len(cnx)))
class Mesh:
    NN = len(x)
    NC = len(cnx)
    Nodos = x
    Conex = cnx

# fig, ax = plt.subplots(figsize=(30,7),dpi = 200)
# graph(Mesh.Nodos,Mesh.Conex,ax,logo=False) # labels=True,d=0.001)
# plt.show()

BC_data = []
for i in range(8,14): 
  BC_data.append([i,1,1,0.0]); BC_data.append([i,1,2,0.0])
for i in range(204,246): 
  BC_data.append([i,1,1,0.0]); BC_data.append([i,1,2,0.0])
for i in range(69,95): 
  BC_data.append([i,0,2,-2*kgf])
BC_data = array(BC_data,'float')


#################              PROCESAMIENTO            ####################
N = Mesh.NN*ElementData.dof
u = zeros(N,'float64')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)#,showBC=True)
# Resuelve el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Solver demoró %.4f segundos"%(time.time()-start))

print('Desplazamiento en X (cm): %.4f'%(u[69*2+1]*100))


#################               ANÁLISIS MODAL              ####################
M = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MasaConcentrada')
# print(M.todense())
Kr = K_reduce(K,BC_data,ElementData)
Mr = K_reduce(M,BC_data,ElementData)
NM = 5

print("Obteniendo valores y vectores propios...")
start = time.time()
vals, vecs = eigsh(Kr, M=Mr, k=NM, which='SM',tol=1E-6)
print("eigsh demoró %.4f segundos"%(time.time()-start))

for i in range(NM):
  texto = 'Modo %i, Frecuencia: %8.2f Hz'%(i+1,vals[i]**0.5/(2*pi))
  mode = V_insert(vecs[:,i],BC_data,ElementData)
  defo = deform(Mesh.Nodos,mode,FS=0.01)
  print(texto)
  fig, ax = plt.subplots(figsize=(20,5))
  graph(defo,Mesh.Conex,ax,logo=False) # labels=True,d=0.001)
  plt.show()

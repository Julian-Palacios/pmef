from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, plot2D_deform, stress

import time
from numpy import array, zeros, append, all
from scipy.sparse.linalg import spsolve
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
    E  = 2.1e10
    v = 0.2
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

BC_data = []
for i in range(8,14): 
  BC_data.append([i,1,1,0.0]); BC_data.append([i,1,2,0.0])
for i in range(204,246): 
  BC_data.append([i,1,1,0.0]); BC_data.append([i,1,2,0.0])
for i in range(69,95): 
  BC_data.append([i,0,2,-5.0])
BC_data = array(BC_data,'float')

N = Mesh.NN*ElementData.dof
u = zeros(N,'float64')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')

K = K.todense()
for i in range(len(K)):
    if all((K[i]==0.0)): print('Hey! K[%i] = 0'%i)

f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)#,showBC=True)
# Resuelve el sistema de ecuaciones
start = time.time()
u = solve(K,f)
print("Solver demoró %.4f segundos"%(time.time()-start))

print('Desplazamiento en X (cm): %.4f'%(u[69*2]*100))
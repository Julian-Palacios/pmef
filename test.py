import time
import matplotlib.pyplot as plt
from numpy import array, append, zeros
from scipy.sparse.linalg import spsolve

from pmef.pre import BC_2Dy
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import graph, deform, plot2D_deform, stress

cm = 0.01
kgf = 9.81
tonf = 1000*kgf

gravity = array([0.0,-1.0,0.0])
class ProblemData:
	SpaceDim = 2
	pde="Elasticity"
class ModelData:
	E  = 2.5e10
	v = 0.25
	thickness = 0.5
	state = 'PlaneStrain'
	density = 2400
	selfweight= 0.0
	gravity = gravity
Lx,Ly = 10.0,0.5

# Usando elementos Tri3
class ElementData:
	dof = 2
	nodes = 3
	noInt =  1
	type = 'Tri3'

def gen2Dpoints(Lx,Ly,ne):
  if Lx>Ly: ms = Ly/ne
  else: ms = Lx/ne
  print(ms)
  nx, ny = int(Lx/ms)+1, int(Ly/ms)+1
  coor = zeros((nx*ny,2),'float')
  k = 0
  for j in range(ny):
    for i in range(nx):
      coor[k] = [i*ms,j*ms]
      k = k+1
  return coor

x = gen2Dpoints(Lx,Ly,4)

from scipy.spatial import Delaunay
from pmef.pre import delaunay
# cnx = Delaunay(x).simplices
cnx = delaunay(x)
# fig, ax = plt.subplots(figsize=(15,1),dpi=200)
# graph(x,cnx,ax,labels=True)
# plt.show()

class Mesh:
  NN = len(x)
  Nodos = x
  NC = len(cnx)
  Conex = cnx
print('NN,NC:',Mesh.NN,Mesh.NC)
#


# Usando Elementos Quad
# class ElementData:
# 	dof = 2
# 	nodes = 4
# 	noInt =  4
# 	type = 'Quad4'
# from pmef.pre import GenQuadMesh_2D
# Mesh = GenQuadMesh_2D(Lx,Ly,4)
# fig, ax = plt.subplots(figsize=(15,1))
# graph(Mesh.Nodos,Mesh.Conex,ax,labels=True)
# plt.show()
#

BC_data = BC_2Dy(Mesh.Nodos, 0.0, [0.0,0.5], tipo=1, gdl=[1,2], val=0.0)
BC = BC_2Dy(Mesh.Nodos, Lx, [0.0,0.5], tipo=0, gdl=[2], val=-1e5)
BC_data = append(BC_data, BC, axis=0)


print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
# Resuelve el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Solver demoró %.4f segundos"%(time.time()-start))

for i in range(len(Mesh.Nodos)):
  if Mesh.Nodos[i,0]==Lx and Mesh.Nodos[i,1]==0.0:
    print('Desplazamiento en X (cm): %.4f'%(u[i*2+1]*100))

print("Generando gráfica...")
FS = 20 # Factor para visualizacion
defo = deform(Mesh.Nodos,u,FS)

fig, ax = plt.subplots(figsize=(15,6))
u_plot = u[1::2]/cm # u para el ploteo
plot2D_deform(u_plot,defo,Mesh.Conex,ax,bar_label='Desplazamiento Y (cm)')


sig = stress(u, Mesh, ElementData, ProblemData, ModelData)
j = 0 # sxx
defo = deform(Mesh.Nodos,u,FS=20)
fig, ax = plt.subplots(figsize=(15,6),dpi=200)
plot2D_deform(sig[:,j]/(kgf/cm**2),defo,Mesh.Conex,ax,bar_label='Sxx (kg/cm2)')
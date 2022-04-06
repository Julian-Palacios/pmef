from typing import TextIO
from pmef.pre import GenBrickMesh_3D
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, graph, K_reduce, V_insert

from scipy.sparse.linalg import spsolve
from numpy import array, zeros, pi
import matplotlib.pyplot as plt
import time


gravity = array([0.0,0.0,-1.0])
class ProblemData:
	SpaceDim = 3
	pde="Elasticity"
class ModelData:
    E  = 25e9
    v = 0.25
    density = 2400
    selfweight= 0.0 # 2400*9.80665
    gravity = gravity
class ElementData:
	dof = 3
	nodes = 8
	noInt =  8
	type = 'Brick8'


#################             PREPROCESAMIENTO            ####################
L, a, ns = 10.0, 0.5, 4
Mesh = GenBrickMesh_3D(L,a,a,ns)
Mesh.Nodos = Mesh.Nodos
print('NN,NC:',Mesh.NN,Mesh.NC)

Pload = -1e5/(ns+1)**2
BC_data = []
for i in range(Mesh.NN):
    [x,y,z] = Mesh.Nodos[i]
    if x == 0.0:
        BC_data.append([i,1,1,0.0])
        BC_data.append([i,1,2,0.0])
        BC_data.append([i,1,3,0.0])
    if x == L:
        BC_data.append([i,0,3,Pload])
BC_data = array(BC_data)
# print(BC_data)

#################               PROCESAMIENTO              ####################
N = Mesh.NN*ElementData.dof
u = zeros(N,'float64')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
f = zeros(N,'float64')
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
# Resuelve el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Solver demoró %.4f segundos"%(time.time()-start))

for i in range(Mesh.NN):
  if Mesh.Nodos[i,0]==L and Mesh.Nodos[i,1]==a/2 and Mesh.Nodos[i,2]==a/2:
    print('Disp. of Node %i at (%.4f,%.4f,%.4f):'%(i,Mesh.Nodos[i,0],Mesh.Nodos[i,1],Mesh.Nodos[i,2]),u[3*i:3*(i+1)])


#################              POSTPROCESAMIENTO            ####################
defo = deform(Mesh.Nodos,u,FS=10.0)
fig = plt.figure(figsize=(8,8),dpi=100)
ax = fig.add_subplot(111,projection='3d')
graph(Mesh.Nodos,Mesh.Conex,ax,color='k',logo=False)
graph(defo,Mesh.Conex,ax,color='r',logo=False)
plt.tight_layout(); plt.show()


#################               ANÁLISIS MODAL              ####################
# from scipy.sparse.linalg import eigsh

# M = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MasaConcentrada')
# Mr = K_reduce(M, BC_data, ElementData)
# Kr = K_reduce(K, BC_data, ElementData)
# # print(Mr.todense())

# NM = 6
# print('Obteniendo Valores y Vectores propios...')
# start = time.time()
# vals, vecs = eigsh(Kr, M=Mr, k=NM, which='SM',tol=1E-6)
# print('eigsh demoró: %.3f segundos.'%(time.time()-start))

# for i in range(NM):
#     texto = 'Modo %i, Frecuencia: %8.2f Hz'%(i+1,vals[i]**0.5/(2*pi))
#     print(texto)
#     # mode = V_insert(vecs[:,i],BC_data, ElementData)
#     # defo = deform(Mesh.Nodos,mode,FS=100)
#     # fig = plt.figure(figsize=(8,8),dpi=100)
#     # ax = fig.add_subplot(111,projection='3d')
#     # graph(Mesh.Nodos,Mesh.Conex,ax,color='k',logo=False)
#     # graph(defo,Mesh.Conex,ax,color='r',logo=False)
#     # plt.title(texto); plt.tight_layout(); plt.show()


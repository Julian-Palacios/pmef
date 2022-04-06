from pmef.pre import LinearMesh
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, K_reduce, V_insert

import time
from numpy import array, zeros, pi
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


class ProblemData:
	SpaceDim = 1
	pde = "Bernoulli"
class ElementData:
	dof = 2
	nodes = 2
	noInt =  2
	type = 'BarB'
class MassData:
	dof = 2
	nodes = 2
	noInt =  4
	type = 'BarB'
E, G, b, h = 25e9, 10e9, 0.5,0.5
gravity = array([0.0,-1.0,0.0])
class ModelData:
	EI  = E*b*h**3/12.0
	GAs = 0.0
	Area=b*h
	fy=0.0#-12000.0
	density = 2400
	gravity = gravity
print('EI:',ModelData.EI/1e9)
print('GAs:',ModelData.GAs/1e9)
#################             PREPROCESAMIENTO            ####################
Ne = 5
Mesh = LinearMesh(10.0,Ne)
# print(Mesh.Nodos,'\n',Mesh.Conex)
BC_data = array([[0,1,1,0.0],[0,1,2,0.0],[Ne,0,2,-1e5]],'float')

#################               PROCESAMIENTO              ####################
N = Mesh.NN*ElementData.dof
u = zeros(N,'float')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
# Resuelve el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Solver demoró %.4f segundos"%(time.time()-start))
print('u en x=L (cm):',u[-1]*100)

#################              POSTPROCESAMIENTO            ####################
print("Generando gráfica...")
FS = 100# Factor para visualizacion
defo = deform(Mesh.Nodos,u[1::2],FS)

plt.figure(figsize=(6,2),dpi=200)
u_plot = u[1::2]/100. # u para el ploteo

plt.plot(Mesh.Nodos,zeros(Mesh.NN),'ko-',label='Posición Original')
plt.plot(Mesh.Nodos,u_plot,'ro:',markersize=4.0,label='$u$ (FEM)')
plt.ylabel('Deformación (cm)')
plt.legend(); plt.show()

#################               ANÁLISIS MODAL              ####################
M = AssembleMatrix(Mesh, MassData, ProblemData, ModelData, 'MasaConcentrada')
# print(M.todense())
Kr = K_reduce(K,BC_data,ElementData)
Mr = K_reduce(M,BC_data,ElementData)
NM = 5
vals, vecs = eigsh(Kr, M=Mr, k=NM, which='SM')

i = 0
plt.figure(figsize=(8,5),dpi=150)
texto = 'Modo %i, Frecuencia: %8.2f Hz'%(i+1,vals[i]**0.5/(2*pi))
mode = V_insert(vecs[:,i],BC_data,ElementData)
plt.plot(mode[1::2],label=texto)
plt.legend(); plt.show()

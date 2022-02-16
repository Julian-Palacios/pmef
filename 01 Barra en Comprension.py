from pmef.pre import LinearMesh
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, graph

import time
from numpy import array, zeros, append
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# Unidades
m = 1
cm = 0.01*m


class ProblemData:
	SpaceDim = 1
	pde="Elasticity"
class ModelData:
	E  = 2e11
	area = 250*cm**2
	density = 7860
	selfweight= 22e4/(4.5*250*cm**2)
	gravity = -1
class ElementData:
	dof = 1
	nodes = 2
	noInt =  1
	type = 'Bar1'


#################             PREPROCESAMIENTO            ####################
Ne = 4
Mesh = LinearMesh(18.0,Ne)
# print(Mesh.Nodos,'\n',Mesh.Conex)

h = -19e4 # F
BC_data = array([[0,1,1,0.0], # Nodo, Tipo de BC, gdl, Valor
                 [Ne,0,1,h]],'float')

#################               PROCESAMIENTO              ####################
N = Mesh.NN*ElementData.dof
u = zeros(N,'float64')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
# Resuelve el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Solver demoró %.4f segundos"%(time.time()-start))
print('u en x=L (cm):',u[Ne]*100)

#################              POSTPROCESAMIENTO            ####################
print("Generando gráfica...")
FS = 100# Factor para visualizacion
defo = deform(Mesh.Nodos,u,FS)

plt.figure(figsize=(6,2),dpi=200)
u_plot = u/cm # u para el ploteo

plt.plot(Mesh.Nodos,zeros(Mesh.NN),'ko-',label='Posición Original')
plt.plot(defo,zeros(Mesh.NN),'ro:',markersize=4.0,label='Deformada $FS=100$')
plt.plot(Mesh.Nodos,u_plot,'b--',lw=0.7,label='$u$ (FEM)')
plt.ylim([-.4,0.1]); plt.ylabel('Deformación Axial (cm)')
plt.legend(); plt.show()

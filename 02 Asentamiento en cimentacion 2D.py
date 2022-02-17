from pmef.pre import delaunay, founMesh, BC_2Dx, BC_2Dy
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, plot2D_deform

import time
from numpy import array, zeros, append
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Unidades
cm = 0.01
kgf = 9.81
tonf = 1000*kgf

gravity = array([0.0,-1.0,0.0])
class ProblemData:
	SpaceDim = 2
	pde="Elasticity"
class ElementData:
	dof = 2
	nodes = 3
	noInt =  1
	type = 'Tri3'
class ModelData:
	E  = 1e7
	v = 0.33
	thickness = 1.0
	state = 'PlaneStress'
	density = 1631
	selfweight= 0.0
	gravity = gravity


D = 1.
mz1 = 0.1
nd = int(D/mz1)
x = zeros((nd+1,2))
for i in range(nd+1):
    x[i] = [mz1*i-D/2,0.]
Lx1, Ly1 = D/2, 0.
Lx2, Ly2 = D, D/2
coor = founMesh(Lx1,Ly1,Lx2,Ly2,mz1)
x = append(x,coor,axis=0)

mz2 = 0.2
Lx1, Ly1 = Lx2, Ly2
Lx2, Ly2 = 2*D, 2*D
coor = founMesh(Lx1,Ly1,Lx2,Ly2,mz2)
x = append(x,coor,axis=0)

mz3 = 0.4
Lx1, Ly1 = Lx2, Ly2
Lx2, Ly2 = 4*D, 4*D
coor = founMesh(Lx1,Ly1,Lx2,Ly2,mz3)

x = append(x,coor,axis=0)
cnx=delaunay(x)

class Mesh:
    NN = len(x)
    NC = len(cnx)
    Nodos = x
    Conex = cnx

#################             PREPROCESAMIENTO            ####################
# BC_data = PRE.BC_2Dx(Mesh.Nodos,dy=0.0,x=[-D/2,D/2],tipo=0,gdl=[2],val=-1*9810.)
BC_data = BC_2Dx(Mesh.Nodos,dy=0.0,x=[-D/2,D/2],tipo=1,gdl=[2],val=-0.01)
BC = BC_2Dx(Mesh.Nodos,dy=-4*D,x=[-4*D,4*D],tipo=1,gdl=[1,2],val=0.)
BC_data = append(BC_data,BC,axis=0)
BC = BC_2Dy(Mesh.Nodos,dx=-4*D,y=[-4*D,0.],tipo=1,gdl=[1],val=0.)
BC_data = append(BC_data,BC,axis=0)
BC = BC_2Dy(Mesh.Nodos,dx=4*D,y=[-4*D,0.],tipo=1,gdl=[1],val=0.)
BC_data = append(BC_data,BC,axis=0)

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

#################              POSTPROCESAMIENTO            ####################
print("Generando gráfica...")
FS = 10 # Factor de Amplificación para deformada
defo = deform(Mesh.Nodos,u,FS)

fig, ax = plt.subplots(figsize=(15,6),dpi=100)
ax.plot(defo[:nd+1,0],defo[:nd+1,1],'k-',lw=4)
u_plot = u[1::2]/cm # u para ploteo a color
plot2D_deform(u_plot,defo,Mesh.Conex,ax,color='RdYlGn',bar_label='Desplazamiento X (cm)')

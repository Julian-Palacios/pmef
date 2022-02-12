from pmef.pre import delaunay, founMesh, BC_2Dx, BC_2Dy
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import Deformada, graph

import time
from numpy import array, zeros, append
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as tri
import matplotlib.cm

gravity = array([0.0,-1.0,0.0])
class ProblemData:
	SpaceDim = 2
	pde="Elasticidad"
class ElementData:
	dof = 2
	nodes = 3
	noInt =  1
	type = 'Tri3'
class ModelData:
	E  = 1e7
	v = 0.33
	thickness = 1.0
	density = 1631
	selfweight= 0.0
	gravity = gravity



D = 1.
mz1 = 0.1
nd = int(D/mz1)
x = zeros((nd+1,2))
for i in range(nd+1):
    x[i] = [mz1*i-D/2,0.]
fig, ax = plt.subplots()
ax.plot(x[:,0],x[:,1],'k-',lw=4,alpha=0.5)
x1, y1 = D/2, 0.
x2, y2 = D, D/2
coor = founMesh(x1,y1,x2,y2,mz1)
x = append(x,coor,axis=0)

mz2 = 0.2
x1, y1 = x2, y2
x2, y2 = 2*D, 2*D
coor = founMesh(x1,y1,x2,y2,mz2)
x = append(x,coor,axis=0)

mz3 = 0.4
x1, y1 = x2, y2
x2, y2 = 4*D, 4*D
coor = founMesh(x1,y1,x2,y2,mz3)
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
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
# Resolviendo el sistema de ecuaciones
start = time.time()
u = spsolve(K.tocsr(),f)
print("Demoró %.4f segundos"%(time.time()-start),"\nCantidad de EF:",Mesh.NC)

#################              POSTPROCESAMIENTO            ####################
FS = 10 # Factor de Amplificación
defo = Deformada(Mesh.Nodos,u,FS)
ax.plot(defo[:nd+1,0],defo[:nd+1,1],'k-',lw=4)
color = "RdYlGn"
up = u[1::2]*100. # u to plot
triangulation = tri.Triangulation(defo[:,0], defo[:,1], triangles=cnx)
ax.tricontourf(triangulation, up, cmap=color, alpha=1.0)
FS = 1.
norm = colors.Normalize(vmin=min(up / FS), vmax=max(up / FS))
plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color),
                orientation='vertical',label='Desplazamiento Y (cm)')
graph(defo,cnx,ax)
plt.axis('off')
plt.axis('equal')
plt.show()

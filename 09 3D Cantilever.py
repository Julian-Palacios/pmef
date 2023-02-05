# 1.1 Importa librerías
from pmef.pre import GenBrickMesh_3D
from pmef.pro import AssembleMatrix,AssembleVector,ApplyBC
from pmef.pos import deform

from numpy import array
from numpy.linalg import norm
from time import time
from scipy.sparse.linalg import spsolve

import meshio


# 1.2 Define el problema
gravity = array([0.0,0.0,-1.0])
class ProblemData:
    SpaceDim = 3
    pde="Elasticity"
class ModelData:
    E  = 2e11
    v = 0.3
    density = 7860
    selfweight= 0.0 # No peso propio
    gravity = gravity
class ElementData:
    dof = 3
    nodes = 8
    noInt = 8
    type = 'Brick8'

# Define unidades
mm = 0.001    # metro
kgf = 9.80665 # Newton


# 1.3 Crea el mallado
Lx, Ly, Lz = 8*mm, 8*mm, 200*mm
ns = 8
Mesh = GenBrickMesh_3D(Lx,Ly,Lz,ns)
n = Mesh.NN
nxy = (ns+1)*(ns+1)
f = -10*kgf
fi= f/nxy 
print("Grados de libertad:",3*n)
print("Nodos en plano XY:",nxy)

# Exporta el mallado creado
mesh = meshio.Mesh(Mesh.Nodos,[("hexahedron",Mesh.Conex)])
mesh.write('output/mesh.vtk')


# 1.4 Crea las Condiciones de Borde
BC_data = []
for i in range(0,nxy): 
    BC_data.append([i,1,1,0.0])
    BC_data.append([i,1,2,0.0])
    BC_data.append([i,1,3,0.0])
for i in range(n-nxy,n):
    BC_data.append([i,0,1,fi])
BC_data = array(BC_data,'float')


# 2.1 Realiza el Ensamble y aplicamos Condiicones de Borde
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK',showTime=True)
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF",showTime=True)
[K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData,ProblemData,ModelData,showTime=True)


# 2.2 Resuelve el sistema de ecuaciones
print("\nResolviendo el sistema...")
start = time()
u = spsolve(K,f)
print("Scipy solver demoró %.4f segundos"%(time()-start))
print(u[-3])
err = (norm(f-K@u)/norm(f))
print("error:%.4e"%err)


# 3 Exporta los resultados
defo = deform(Mesh.Nodos,u,FS=10.0)
mesh = meshio.Mesh(defo,[("hexahedron",Mesh.Conex)],point_data={"Z":u[0::3]})
mesh.write('output/result.vtk')
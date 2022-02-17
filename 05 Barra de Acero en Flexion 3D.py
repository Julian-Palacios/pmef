from pmef.pre import GenBrickMesh_3D
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC

from scipy.sparse.linalg import spsolve
from numpy import array, zeros
import matplotlib.pyplot as plt
import time


gravity = array([0.0,0.0,-1.0])
class ProblemData:
	SpaceDim = 3
	pde="Elasticity"
class ModelData:
    E  = 2e11
    v = 0.3
    density = 7860
    selfweight= 0.0 # 7860*9.80665
    gravity = gravity
class ElementData:
	dof = 3
	nodes = 8
	noInt =  8
	type = 'Brick8'


#################             PREPROCESAMIENTO            ####################
L = 1.0
a = 0.05
Mesh = GenBrickMesh_3D(L,a,a,2)
Mesh.Nodos = Mesh.Nodos
# print(Mesh.Nodos,'\n',Mesh.Conex)

BC_data = []
for i in range(Mesh.NN):
    [x,y,z] = Mesh.Nodos[i]
    if x == 0.0:
        BC_data.append([i,1,1,0.0])
        BC_data.append([i,1,2,0.0])
        BC_data.append([i,1,3,0.0])
    # else:
        # BC_data.append([i,1,1,0.0])
        # BC_data.append([i,1,2,0.0])
    if x == L:
        BC_data.append([i,0,3,-1e5])
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

#################              POSTPROCESAMIENTO            ####################
# fig = plt.figure(figsize=(8,8))#,dpi=200)
# ax = fig.add_subplot(111,projection='3d')
# ax.plot([0,L],[-L/2,L/2],[-L/2,L/2],alpha=0.0)
# ax.plot(Mesh.Nodos[:,0],Mesh.Nodos[:,1],Mesh.Nodos[:,2],'ko',markersize=3.0)
# ax.plot(Mesh.Nodos[:,0]+u[0::3],Mesh.Nodos[:,1]+u[1::3],Mesh.Nodos[:,2]+u[2::3],'ro',markersize=3.0)
# plt.show()

# for i in range(Mesh.NN):
#     if Mesh.Nodos[i,0]==L:
#         print('Disp. of Node %i at (%.4f,%.4f,%.4f):'%(i,
#                 Mesh.Nodos[i,0],Mesh.Nodos[i,1],Mesh.Nodos[i,2]),u[i*3:(i+1)*3])
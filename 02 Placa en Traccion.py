from pmef.pre import load_obj, BC_2Dy
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import Deformada, plot_deform

import time
from numpy import array, zeros, append
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Unidades
cm = 0.01
kgf = 9.81
tonf = 1000*kgf

# Lectura de Mesh
# Caso 1
Qvertices, Qtri, Qquad = load_obj("sample_input/Quad_mesh.obj")
Qvertices = Qvertices[:,[True,True,False]]*cm
cnx = Qquad - 1
x = Qvertices
class ElementData:
	dof = 2
	nodes = 4
	noInt =  4
	type = 'Quad4'

# Caso 2
# Tvertices, Ttri, Tquad = load_obj("sample_input/Tri_mesh.obj")
# Tvertices = Tvertices[:,[True,True,False]]*cm
# cnx = Ttri - 1
# x = Tvertices
# class ElementData:
# 	dof = 2
# 	nodes = 3
# 	noInt =  1
# 	type = 'Tri3'

print('# Nodos: %i \n# Elementos: %i'%(len(x),len(cnx)))
class Mesh:
    NN = len(x)
    NC = len(cnx)
    Nodos = x
    Conex = cnx


gravity = array([0.0,-1.0,0.0])
class ProblemData:
	SpaceDim = 2
	pde="Elasticidad"
class ModelData:
	E  = 2.1e10 
	v = 0.2
	thickness = 0.04
	state = 'PlainStress'
	density = 7860
	selfweight= 0.0
	gravity = gravity


#################             PREPROCESAMIENTO            ####################
BC_data = BC_2Dy(Mesh.Nodos,dx=0.0,y=[0.1,0.4],tipo=1,gdl=[1,2],val=0.0)
BC = BC_2Dy(Mesh.Nodos,dx=1.0,y=[0.0,0.5],tipo=0,gdl=[1],val=20*tonf)
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
FS = 50# Factor para visualizacion
defo = Deformada(Mesh.Nodos,u,FS)

fig, ax = plt.subplots(figsize=(15,6),dpi=200)
u_plot = u[0::2]/cm # u para el ploteo
plot_deform(u_plot,defo,cnx,ax,bar_label='Desplazamiento X (cm)')

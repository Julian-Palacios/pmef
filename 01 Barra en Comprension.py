from pmef.pre import LinearMesh
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, plot_deform

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
BC_data = array([[0,1,0.0], # Nodo, Tipo de BC, Valor
                 [Ne,0,h]],'float')

#################               PROCESAMIENTO              ####################
N = Mesh.NN*ElementData.dof
u = zeros(N,'float64')
print("Ensamble en el Sistema global...")
K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")


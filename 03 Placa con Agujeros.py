from pmef.pre import BC_2Dy, BC_2Dx
from pmef.pro import AssembleMatrix, AssembleVector, ApplyBC
from pmef.pos import deform, plot_deform

import time
from numpy import array, zeros, append
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np

class Viga():

  def __init__(self,altura,largo,dz,dx):
    self.L = largo
    self.H = altura
    self.dx = dx
    self.dz = dz

    self.X,self.Z = np.meshgrid(np.linspace(0,self.L,self.dx),np.linspace(0,self.H,self.dz))
    self.X = np.reshape(self.X,(np.size(self.X),1))
    self.Z = np.reshape(self.Z,(np.size(self.Z),1))

    self.coordenadas = np.hstack((self.X,self.Z))
    self.centros = []
  
  def add_hole(self,X0,Z0,r,dr,dθ):
    
    for i in range(len(self.coordenadas)):
      if ((self.coordenadas[i][0]-X0)**2+(self.coordenadas[i][1]-Z0)**2)**0.5<(r+(self.H/2-r)*2):
        self.coordenadas[i][0]=np.nan
        self.coordenadas[i][1]=np.nan

    teta = np.linspace(0,2*np.pi-2*np.pi/dθ,dθ)
    r0 = np.logspace(np.log(r),np.log((r+(self.H/2-r)*1.8)),dr,base=np.e)

    Xc = np.empty(dθ*dr)
    Zc = np.empty(dθ*dr)

    i=0
    for radio in r0:
      Xc[i:i+dθ] = radio*np.cos(teta)+X0
      Zc[i:i+dθ] = radio*np.sin(teta)+Z0
      i=i+dθ
    Xc = np.reshape(Xc,(np.size(Xc),1))
    Zc = np.reshape(Zc,(np.size(Zc),1))
    coordenadas_cir = np.hstack((Xc,Zc))
    centro = [[X0,Z0]]
    self.centros = self.centros + centro
    self.coordenadas = np.vstack((self.coordenadas,coordenadas_cir))

    for i in range(len(self.coordenadas)):
      if self.coordenadas[i][0]>=self.L or self.coordenadas[i][1]>=self.H:
        self.coordenadas[i][0]=np.nan
        self.coordenadas[i][1]=np.nan

      elif self.coordenadas[i][0]<=0 or self.coordenadas[i][1]<=0:
        self.coordenadas[i][0]=np.nan
        self.coordenadas[i][1]=np.nan
  
    self.Xb,self.Zb = np.meshgrid(np.linspace(0,self.L,self.dx),np.linspace(0,self.H,self.dz))
    self.Xb = np.reshape(self.X,(np.size(self.X),1))
    self.Zb = np.reshape(self.Z,(np.size(self.Z),1))
    self.coordenadas_borde = np.hstack((self.Xb,self.Zb))

    for i in range(len(self.coordenadas_borde)):
      if self.coordenadas_borde[i][0]<self.L and self.coordenadas_borde[i][0]>0 and self.coordenadas_borde[i][1]>0 and self.coordenadas_borde[i][1]<self.H :
        self.coordenadas_borde[i][0]=np.nan
        self.coordenadas_borde[i][1]=np.nan
    self.coordenadas = np.vstack((self.coordenadas,self.coordenadas_borde))

  def show(self,dh,db):
    fig1=plt.figure(figsize=(dh,db))
    plt.scatter(self.coordenadas[:,0],self.coordenadas[:,1])
    plt.show()

  def triangulation(self,dh,db):
    self.centros = np.array(self.centros)
    self.coordenadas = self.coordenadas[~np.isnan(self.coordenadas).any(axis=1)]
    self.coordenadas = np.vstack((self.coordenadas,self.centros))

    self.tri=Delaunay(self.coordenadas)
    centrosE = [len(self.coordenadas)-1-x for x in range(len(self.centros))]

    selec = []
    for ce in centrosE:
      for ie in range(len(self.tri.simplices)):
        for nodo in self.tri.simplices[ie]:
          if nodo == ce:
            selec.append(ie)
            break

    self.tri.simplices = np.delete(self.tri.simplices, selec, axis=0)
    self.coordenadas = self.coordenadas[:-len(self.centros)]

    fig=plt.figure(figsize=(dh,db))
    for a, b, c in self.tri.simplices:
        for i, j in [(a, b), (b, c), (c, a)]:
            plt.plot(self.coordenadas[[i, j], 0], self.coordenadas[[i, j], 1], color='k')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

    return self.tri.simplices

# Unidades
cm = 0.01
kgf = 9.80665
tonf = 1000*kgf

# Parámetros para Discretización
h, L = 0.5, 4.0
# Caso 0
# Nx = 20
# Viga1=Viga(h,L,5,Nx)
# Viga1.add_hole(0.1*L,h/2,0.15,3,10) 
# Viga1.add_hole(0.3*L,h/2,0.15,3,10)
# Viga1.add_hole(0.5*L,h/2,0.15,3,10)
# Viga1.add_hole(0.7*L,h/2,0.15,3,10)
# Viga1.add_hole(0.9*L,h/2,0.15,3,10)

# Caso 1
# Nx = 32
# Viga1=Viga(h,L,5,Nx)
# Viga1.add_hole(0.1*L,h/2,0.15,3,15) 
# Viga1.add_hole(0.3*L,h/2,0.15,3,15)
# Viga1.add_hole(0.5*L,h/2,0.15,3,15)
# Viga1.add_hole(0.7*L,h/2,0.15,3,15)
# Viga1.add_hole(0.9*L,h/2,0.15,3,15)

# Caso 2
Nx = 64
Viga1=Viga(h,L,10,Nx)
Viga1.add_hole(0.1*L,h/2,0.15,5,30) 
Viga1.add_hole(0.3*L,h/2,0.15,5,30)
Viga1.add_hole(0.5*L,h/2,0.15,5,30)
Viga1.add_hole(0.7*L,h/2,0.15,5,30)
Viga1.add_hole(0.9*L,h/2,0.15,5,30)

# Caso 3
# Nx = 100
# Viga1=Viga(h,L,16,Nx)
# Viga1.add_hole(0.1*L,h/2,0.15,8,45) 
# Viga1.add_hole(0.3*L,h/2,0.15,8,45)
# Viga1.add_hole(0.5*L,h/2,0.15,8,45)
# Viga1.add_hole(0.7*L,h/2,0.15,8,45)
# Viga1.add_hole(0.9*L,h/2,0.15,8,45)

cnx = Viga1.triangulation(16,2.5)
x = Viga1.coordenadas
print('# Nodos: %i \n# Elementos: %i'%(len(x),len(cnx)))

class Mesh:
    NN = len(x)
    NC = len(cnx)
    Nodos = x
    Conex = cnx

gravity = np.array([0.0,-1.0,0.0])
class ProblemData:
	SpaceDim = 2
	pde="Elasticidad"
class ElementData:
	dof = 2
	nodes = 3
	noInt =  1
	type = 'Tri3'
class ModelData:
    E  = 2.1e10 # 1000
    v = 0.2
    thickness = 0.04
    state = 'PlaneStress'
    density = 7860
    selfweight= 0.0
    gravity = gravity


#################             PREPROCESAMIENTO            ####################
P = 5.0*9810.0
BC_data = BC_2Dy(Mesh.Nodos,dx=0,y=[0.0,h],tipo=1,gdl=[1,2],val=0.)
BC = BC_2Dx(Mesh.Nodos,dy=h,x=[0.0,L],tipo=0,gdl=[2],val=-P)
BC_data = append(BC_data,BC,axis=0)
print(sum(sum(BC_data)))

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
FS = 10# Factor para visualizacion
defo = deform(Mesh.Nodos,u,FS)

fig, ax = plt.subplots(figsize=(15,6),dpi=200)
u_plot = u[1::2]/cm # u para el ploteo
plot_deform(u_plot,defo,cnx,ax,bar_label='Desplazamiento X (cm)')

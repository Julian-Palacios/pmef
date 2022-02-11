## Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe) 
## Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
## Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##

from FEM_funciones import *
from DIN_funciones import *
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh

################# DATOS DE ENTRADA ###################
class ProblemData:
	SpaceDim = 1
	pde = "Timoshenko"
class ElementData:
	dof = 2
	nodes = 2
	noInt =  1
	type = 'Bar1'
class MassData:
	dof = 2
	nodes = 2
	noInt =  2
	type = 'Bar1'
E, G, b, h =25e9, 10e9, 0.5,0.5
gravity=np.array([0.0,0.0,0.0])
class ModelData:
	EI  = E*b*h**3/12.0
	GAs=G*5*b*h/6.0
	Area=b*h
	fy=0.0#-12000.0
	density = 2400
	gravity[1] = -1.0
	gravity = gravity

############  MÉTODO DE ELEMENTOS FINITOS  #############
despV,VPs,resp=[],[],[]
L=10## Longitud del elemento
for j in [1,2,5,10,20,50,100]:#Resuelve para 1, 2, ..., 100 Elementos Finitos
  ## Obteniendo Mesh
  Nodos,Conex=LinearMesh(L,j)
  class Mesh:
    NN = len(Nodos)
    Nodos = Nodos.T
    NC = len(Conex)
    Conex = Conex.T
  print('Nodos:\t',Mesh.NN,'\tElementos:\t',Mesh.NC,'\n')
  ## Leyendo Condiciones de Borde
  BC_data=np.array([[1,1,1,0.0],[1,1,2,0.0],[Mesh.NN,0,2,-1e5]])
  #---------------------------------------------------------------------------------------
  print('Empezando el procedimiento para la solución de FE')
  ## Definiendo vector de desplazamiento
  N = Mesh.NN*ElementData.dof
  u    = np.zeros(N,np.float64)
  print('Ensamblando matriz K...')
  K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
  print('Ensamblando matriz Masa...')
  M = AssembleMatrix(Mesh, MassData, ProblemData, ModelData, 'MasaConcentrada')
  print('Ensamblando vector F')
  f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
  print('Aplicando Condiciones de Borde\n')
  [K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData, ModelData)
  ## Resolviendo el sistema de ecuaciones
  print('Resolviendo...')
  u =spsolve(K.tocsr(),f)
  print("Desplazamiento vertical en (L,0)=",u[-1])
  ##
  ## Acumulando desplazamiento para una cantidad de elementos
  despV.append([j,u[-1]])
  ### Estimando Modos
  KK, MM = np.array(K.todense()), np.array(M.todense())
  vals, vecs = eigh(KK[2:,2:],MM[2:,2:])
  vp1=vals[0]**0.5/(2*3.1415926535)
  VPs.append([j,vp1])
  print("\nFrecuencia del Modo 1: %s (Hz)"%vp1)
  ##
#### Descomentar para visualizar modos
####NM= 3##NM es numero de modos a visualizar
##for i in range(NM):
##        plt.plot(vecs[:,i][1::2])
##        plt.show()
##
  ############## RESPUESTA DINÁMICA  ###############
  CC=Amortiguamiento(KK[2:,2:],MM[2:,2:],ζ=0.05,tipo_am="Rayleigh")
  ug=np.genfromtxt("./INPUT/Lima66NS15-35_FF.txt")/100
  dt=0.02
  print('\nResolviendo Problema Dinámico...')
  [d,v,a]=MDOF_LTH(KK[2:,2:],MM[2:,2:],CC,ug,dt)
  ## Acumulando respuesta en desplazamientos en el tiempo
  resp.append(d[:,-1])
  ##
#### Descomentar para ver desplazamiento del extremo en el tiempo
##  plt.plot(d[:,-1])
##  plt.show()
##
###### Descomentar para plotear la convergencia a mayor elementos
####despV=np.array(despV)
####plt.plot(despV[:,0],despV[:,1],"o--")
####plt.axis([0,100,-0.26,0.0])
####plt.show()
##
###### Descomentar para plotear la convergencia a la frecuencia fundamental
####VPs=np.array(VPs)
####plt.plot(VPs[:,0],VPs[:,1],"o--")
####plt.axis([0,100,0.0,3.0])
####plt.show()
##
####Descomentar para guardar archivos  
##np.savetxt("./OUTPUT/Timposhenko_puntual_reducido.txt",despV,fmt="%10.7f")
##np.savetxt("./OUTPUT/Timposhenko_modo1_K1M2.txt",VPs,fmt="%10.7f")
##np.savetxt("./OUTPUT/Respuesta_TimoK1M2_Concen_Rayleigh.txt",np.array(resp).T)

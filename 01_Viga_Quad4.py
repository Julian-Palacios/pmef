## Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe) 
## Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
## Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##

from FEM_funciones import *
from DIN_funciones import *
import time
from scipy.sparse.linalg import spsolve, eigsh
import matplotlib.pyplot as plt
from scipy.linalg import eigh

################# DATOS DE ENTRADA ###################
class ProblemData:
	SpaceDim = 2
	pde="Elasticidad"
class ElementData:
	dof = 2
	nodes = 4
	noInt =  4
	type = 'Quad4'
class MassData:
	dof = 2
	nodes = 4
	noInt =  4
	type = 'Quad4'
gravity=np.array([0.0,0.0,0.0])
class ModelData:
	E  = 25e9
	v = 0.25
	thickness = 0.5
	density = 2400
	selfweight= 0.0##2400*20
	gravity[1] = -1.0
	gravity = gravity

############  MÉTODO DE ELEMENTOS FINITOS  #############
despV,VPs,resp=[],[],[]
##
for j in [0.5,1,1.5,2,2.5,3,3.5,4]:#Resuelve para mesh de lc=2/0.5, ..., 2/4
        L, H, NE_X = 10.0, 0.5, j
        lc=2.0/NE_X
        Mesh_File=GenQuadMesh(L,H,lc)
        Mesh=gmsh_read(Mesh_File,ProblemData, ElementData)
        ##
        BC_coord=np.array([[0.0,0.0,1,1,0.0],
                     [0.0,0.0,1,2,0.0],
                     [0.0,0.5,1,1,0.0],
                     [0.0,0.5,1,2,0.0],
                     [10.0,0.5,0,2,-1e5]])
        ##
        BC_data=genBC_2D(BC_coord,Mesh.Nodos.T,lim=0.001)
        ##
        print('Empezando el procedimiento para la solución de FE')
        ## Definiendo vector de desplazamiento
        N = Mesh.NN*ElementData.dof
        u    = np.zeros(N,np.float64)
        print('Ensamblando matriz K...')
        K = AssembleMatrix(Mesh, ElementData, ProblemData, ModelData, 'MatrizK')
        print('Ensamblando matriz Masa...')
        M = AssembleMatrix(Mesh, MassData, ProblemData, ModelData,  'MasaConcentrada')
        print('Ensamblando vector F')
        f = AssembleVector(Mesh, ElementData, ProblemData, ModelData, "VectorF")
        print('Aplicando Condiciones de Borde\n')
        [K, f] = ApplyBC(K, f, BC_data, Mesh, ElementData, ProblemData,  ModelData)
        ## Resolviendo el sistema de ecuaciones
        print('\nResolviendo Problema Estático...')
        u =spsolve(K.tocsr(),f)
        gdl_x=int((BC_data[-1,0]-1)*2+1)
        print("Desplazamiento vertical en (L,0)=",u[gdl_x])
###### Descomentar para plotear deformada
####        X_Def=Deformada(Mesh.Nodos.T,u,FS=1.0)
####        plt.plot(X_Def[:,0],X_Def[:,1],"o")
####        plt.show()
        ## Acumulando desplazamientos para una cantidad de elementos
        despV.append([Mesh.NC,u[gdl_x]])
        ### Estimando Modos
        K1, M1 = np.array(K.todense()),np.array(M.todense())
        K2=K_reductor(K1,[0,1,4,5])
        M2=K_reductor(M1,[0,1,4,5])
        vals, vecs = eigh(K2,M2)
        vp1=vals[0]**0.5/(2*3.1415926535)
        VPs.append([Mesh.NC,vp1])
        print("\nFrecuencia del Modo 1: %s (Hz)"%vp1)
        ##
###### Descomentar para visualizar modos
####NM= 3##NM es numero de modos a visualizar
########for i in range(NM):
########        print("w%s = %7.2f"%(i,vals[i]**0.5))
########        plt.figure(figsize=(15,2))
########        vec=vecs[:,i]
########        vec=V_insertor(vec,[0,1,4,5])
########        Modo=Deformada(Mesh.Nodos.T,vec,FS=1)
########        plt.plot(Modo[:,0],Modo[:,1],"o")
########        plt.axis([-0.05,10.05,-0.05,0.55])
########        plt.show()
##
        ############## RESPUESTA DINÁMICA  ###############
        CC=Amortiguamiento(K2,M2,ζ=0.05,tipo_am="Rayleigh")
        ug=np.genfromtxt("./INPUT/Lima66NS15-35_FF.txt")/100
        dt=0.02
        print('\nResolviendo Problema Dinámico...')
        [d,v,a]=MDOF_LTH(K2,M2,CC,ug,dt)
        ## Acumulando respuesta en desplazamientos en el tiempo
        resp.append(d[:,int((BC_data[-1,0]-1)*2+1)-4])
		#### Descomentar para ver desplazamiento del extremo en el tiempo
##        plt.plot(d[:,int((BC_data[-1,0]-1)*2+1)-4])
##        plt.show()
        ##
###### Descomentar para plotear la convergencia a mayor elementos
####despV=np.array(despV)
####plt.plot(despV[:,0],despV[:,1],"o--")
####plt.axis([0,Mesh.NC,-0.26,0.0])
####plt.show()
##
###### Descomentar para plotear la convergencia a la frecuencia fundamental
####VPs=np.array(VPs)
####plt.plot(VPs[:,0],VPs[:,1],"o--")
####plt.axis([0,Mesh.NC,0.0,4.5])
####plt.show()
##
####Descomentar para guardar archivos
##np.savetxt("./OUTPUT/Viga2D_puntual.txt",despV,fmt="%10.7f")
####np.savetxt("./OUTPUT/Viga2D_Modo1.txt",VPs,fmt="%10.7f")
##np.savetxt("./OUTPUT/Respuesta_2D_Concentrada_Rayleigh.txt",np.array(resp).T)
        





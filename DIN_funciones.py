## Código creado en el año 2020 por Julian Palacios (jpalaciose@uni.pe) 
## Con la colaboración de Jorge Lulimachi (jlulimachis@uni.pe)
## Codificado para la clase FICP-C811 en la UNI, Lima, Perú
##

import numpy as np
from numpy.linalg import det, inv

def Amortiguamiento(K,M,ζ=0.05,tipo_am="Rayleigh"):
  '''Función que Estima la matriz de amortiguamineto según el tipo de
    de método escogido (Rayleigh, Modal, ...)
  '''
  from scipy.linalg import eigh
  vals, vecs=eigh(K,M)
  n=len(M)
  ##
  if tipo_am=="Rayleigh":
    tipo=(type(ζ).__name__)
    if tipo=='float':
      if not 0.0<=ζ or ζ>=1.0: print("Error! El valor de ζ debe estar en [0,1]")
      ζ=np.zeros(2)+ζ
    elif tipo=='list':
      if not len(ζ)==2: print("Error! Definir ζ como una lista: [ζi,ζf]")
    else: print("Error! ζ debe ser decimal o un lista de 2 valores [ζi,ζf]")
    ##
    β=2*(vals[int(n/2)]**0.5*ζ[1]-vals[0]**0.5*ζ[0])/(vals[0]+vals[int(n/2)])
    α=2*vals[0]**0.5*ζ[0]-β*vals[0]
    print("\nw0,w%s:%s,%s\nα,β=%s,%s"%(int(n/2),vals[0]**0.5,vals[int(n/2)]**0.5,α,β))
    C=α*M+β*K
    ##
######Descomentar para plotear amortiguamiento de cada modo
####    for i in range(n):
####      vecsR=np.reshape(vecs[:,i],(1,n))
####      vecsC=np.reshape(vecs[:,i],(n,1))
####      Mn=vecsR@M@vecsC
####      print("w%s=%s\n"%(i,vals[i]**0.5),"ζ%s=%s\n"%(i,vecsR@C@vecsC/(2*Mn*vals[i]**0.5)))
  elif tipo_am=="Modal":
    tipo=(type(ζ).__name__)
    if tipo=='float':
      if not 0.0<=ζ or ζ>=1.0: print("Error! El valor de ζ debe estar en [0,1]")
      ζ=np.zeros(int(n/2))+ζ
    elif tipo=='list':
      if not len(ζ)==n: print("Error! El número de elementos en ζ debe ser %s"%n)
    else: print("Error! ζ debe ser decimal o un lista")
    ##
    C=np.zeros((n,n))
    for i in range(int(n/2)):
      vecsR=np.reshape(vecs[:,i],(1,n))
      vecsC=np.reshape(vecs[:,i],(n,1))
      Mn=vecsR@M@vecsC
      C=C+M@(2*ζ[i]*vals[i]**0.5*vecsC@vecsR/Mn)@M
      ##
######Descomentar para plotear amortiguamiento de cada modo
####    for i in range(int(n/2)):
####      vecsR=np.reshape(vecs[:,i],(1,n))
####      vecsC=np.reshape(vecs[:,i],(n,1))
####      print(vecsR@C@vecsC/(2*Mn*vals[i]**0.5)) 
  else:
    print("Aún no se ha programado para %s"%tipo_am)
  return C

def MDOF_LTH(K,M,C,ug,dt,γ=1/2,β=1/4):
  ''' Función que estima la respuesta dinámica lineal de una estructura
      a través del método de newmark usando la formulación incremental.
  '''
  from numpy.linalg import inv
  ##
  n=len(M)
  ns=len(ug)
  dx=np.zeros((ns+1,n))
  d=np.zeros((ns+1,n))
  v=np.zeros((ns+1,n))
  a=np.zeros((ns+1,n))
  df=np.zeros((ns+1,n))
  ##
  df[0]=0
  df[1]=-(ug[1]-ug[0])
  ##
  c1=1-γ/β
  c2=γ/(β*dt)
  c3=dt*(1-γ/(2*β))
  ##
  a1=M/(β*dt**2)+γ*C/(β*dt)+K
  a2=M/(β*dt)+γ*C/β
  a3=M/(2*β)-dt*(1-γ/(2*β))*C
  I=np.ones(n)
  I[0::2]=0.0
  MI=M@I
  ##
  ## Solución de la ecuación diferencial
  for i in range(1,ns):
    if i%100==0: print("Paso:\t%s\nTiempo:\t%s(s)"%(i,i*dt))
    dx[i]=inv(a1)@(a2@v[i-1]+a3@a[i-1]+MI*df[i])
    ##
    d[i]=d[i-1]+dx[i]
    v[i]=c1*v[i-1]+c2*dx[i]+c3*a[i-1]
    a[i]=-inv(M)@(C@v[i]+K@d[i]+MI*ug[i])
    ##
    df[i]= -(ug[i]-ug[i-1])
  return d,v,a
  

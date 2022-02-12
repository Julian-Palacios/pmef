from numpy import zeros, array, average, append
import matplotlib.pyplot as plt
import matplotlib.image as image


def Deformada(X,u,FS=10.0):
        ''' Función que agrega una deformación a la pocisión de Nodos
        Input:
            X:      Arreglo que contiene las coordenadas de todos los Nodos
            u:      Arreglo que contiene las deformaciones calculadas con el MEF
            FS:     Factor de Escala
        Output:
            X_def:  Arreglo que contiene la deformada del sistema.   
        '''
        NN = len(X)
        X_def=zeros(X.shape)
        for i in range(NN):
            X_def[i]=X[i]+FS*array([u[2*i],u[2*i+1]])
        return X_def

def graph(x,cnx,ax):
    d=0.01
    for ii in range(len(x)):
        ax.plot(x[ii,0],x[ii,1],'ko',markersize=1.5)
        # ax.annotate('%i'%(ii+1),(x[ii,0],x[ii,1]),
        #                 xytext=(x[ii,0]+d,x[ii,1]+d),color='black')
    ie=0
    for e in cnx:
        xx = x[append(e,e[0])]
        ax.plot(xx[:,0],xx[:,1],'k',lw = 0.3,alpha=0.5)
        xa,ya=average(xx[:,0]),average(xx[:,1])
        # ax.annotate('%i'%(ie+1),(xa,ya),color='blue',fontsize=8.)
        ie += 1

    xrng, yrng = plt.xlim(),plt.ylim()
    xm, ym = (xrng[0]+xrng[1])/2, (yrng[0]+yrng[1])/2
    L = min(yrng[1]-xrng[0],yrng[1]-yrng[0])
    im = image.imread('https://jpi-ingenieria.com/images/logoJPI.png')
    ax.imshow(im,aspect='auto',extent=(xm-0.3*L-1*L, xm+0.3*L-1*L, ym-0.3*L-1*L, ym+0.3*L-1*L), zorder=-1,alpha=0.5)

def quads_to_tris(quads):
    tris = zeros((2*len(quads),3),'int')
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i,0]
        n1 = quads[i,1]
        n2 = quads[i,2]
        n3 = quads[i,3]
        tris[j,0] = n0
        tris[j,1] = n1
        tris[j,2] = n2
        tris[j+1,0] = n2
        tris[j+1,1] = n3
        tris[j+1,2] = n0
    return tris
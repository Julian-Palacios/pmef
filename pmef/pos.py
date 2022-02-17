from numpy import zeros, array, average, append
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
import matplotlib.tri as tri
import matplotlib.cm
from sqlalchemy import false


def deform(X,u,FS=10.0):
    ''' Función que agrega una deformación a la pocisión de Nodos.

    Input:
        X:      Arreglo que contiene las coordenadas de todos los Nodos
        u:      Arreglo que contiene las deformaciones calculadas con el MEF
        FS:     Factor de Escala
    Output:
        X_def:  Arreglo que contiene la deformada del sistema.   
    '''
    if len(X.shape)==1:
        dim = 1
        NN = len(X)
    else:
        (NN,dim) = X.shape
    X_def=zeros((NN,dim))
    if dim == 1: # 1D
        X_def = X + u*FS
    elif dim==2: # 2D
        for i in range(NN):
            X_def[i]=X[i]+FS*array([u[2*i],u[2*i+1]])
    elif dim == 3:
        for i in range(NN):
            X_def[i] = X[i]+FS*array([u[3*i],u[3*i+1],u[3*i+2]])
    else:
        print('Debe programar para %iD!'%dim)
    return X_def

def graph(x,cnx,ax,color='k',d=0.01,logo=True,labels=False):
    '''
    Función que grafica los nodos y elementos del modelo.
    '''
    if len(x.shape)==1:
        dim = 1
        NN = len(x)
    else:
        (NN,dim) = x.shape

    if dim == 1:
        for ii in range(len(x)):
            ax.plot(x[ii],0.0,color+'o',markersize=5.0)
            if labels == True:
                ax.annotate('%i'%(ii+1),(x[ii],0.0),
                                xytext=(x[ii]+d,0.0+d),color='black')
        ie=0
        for e in cnx:
            xx = x[e]
            ax.plot(xx,[0.0,0.0],color,lw = 1.0,alpha=1.0)
            if labels == True:
                xa,ya=average(xx),0.0
                ax.annotate('%i'%(ie+1),(xa,ya+d),color='blue',fontsize=8.)
            ie += 1
        Lx = max(x) - min(x)
        plt.axis([average(x)-1.02*Lx/2,average(x)+1.02*Lx/2,-Lx/4,Lx/4])

    elif dim == 2:
        for ii in range(len(x)):
            ax.plot(x[ii,0],x[ii,1],'ko',markersize=1.5)
            if labels == True:
                ax.annotate('%i'%(ii+1),(x[ii,0],x[ii,1]),
                                xytext=(x[ii,0]+d,x[ii,1]+d),color='black')
        ie=0
        for e in cnx:
            xx = x[append(e,e[0])]
            ax.plot(xx[:,0],xx[:,1],'k',lw = 0.3,alpha=0.5)
            if labels == True:
                xa,ya=average(x[e,0]),average(x[e,1])
                ax.annotate('%i'%(ie+1),(xa,ya),color='blue',fontsize=8.)
            ie += 1

    elif dim == 3:
        # print('Hacer algo...')
        for ii in range(len(x)):
            ax.plot(x[ii,0],x[ii,1],x[ii,2],color+'o',markersize=1.5)
            if labels == True:
                ax.text(x[ii,0],x[ii,1],x[ii,2],'%i'%(ii+1),color='black')
        ie=0
        for e in cnx:
            xx = x[[e[0],e[1],e[2],e[3],e[0],e[4],e[5],e[6],e[7],e[4]]]
            ax.plot(xx[:,0],xx[:,1],xx[:,2],color,lw = 0.3,alpha=0.5)
            ax.plot(x[[e[1],e[5]],0],x[[e[1],e[5]],1],x[[e[1],e[5]],2],'k',lw = 0.3,alpha=0.5)
            ax.plot(x[[e[2],e[6]],0],x[[e[2],e[6]],1],x[[e[2],e[6]],2],'k',lw = 0.3,alpha=0.5)
            ax.plot(x[[e[3],e[7]],0],x[[e[3],e[7]],1],x[[e[3],e[7]],2],'k',lw = 0.3,alpha=0.5)
            if labels == True:
                xa,ya,za = average(x[e,0]),average(x[e,1]),average(x[e,2])
                ax.text(xa,ya,za,'%i'%(ie+1),color='blue',fontsize=8.)
            ie += 1
        L = x.max()-x.min()
        ax.plot(average(x[:,0])+L/2,average(x[:,1])+L/2,average(x[:,2])+L/2,alpha=0.0)
        ax.plot(average(x[:,0])-L/2,average(x[:,1])-L/2,average(x[:,2])-L/2,alpha=0.0)
        logo = False

    if logo == True:
        xrng, yrng = plt.xlim(),plt.ylim()
        L = min(yrng[1]-xrng[0],yrng[1]-yrng[0])
        im = image.imread('https://jpi-ingenieria.com/images/logoJPI.png')
        ax.imshow(im,aspect='auto',extent=(xrng[0], xrng[0]+0.2*L, yrng[0], yrng[0]+0.2*L), zorder=10,alpha=0.7)

def quads_to_tris(quads):
    '''
    Función que genera elementos tri a partir de elementos quad.
    '''
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

def plot2D_deform(up,defo,cnx,ax,color='RdYlGn_r',bar_label='Resultados',logo=True,labels=False):
    '''
    Función que realiza el ploteo de nodos, elementos y resultados.
    '''
    if cnx.shape[1]==3: 
        triangulation = tri.Triangulation(defo[:,0], defo[:,1], triangles=cnx)
    elif cnx.shape[1]==4:
        tris = quads_to_tris(cnx)
        triangulation = tri.Triangulation(defo[:,0], defo[:,1], triangles=tris)
    else: print('pmef no soporta el mesh proporcionado para la gráfica.')
    ax.tricontourf(triangulation, up, cmap=color, alpha=1.0)
    norm = colors.Normalize(vmin=min(up), vmax=max(up))
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color),
                    orientation='vertical',label=bar_label)
    graph(defo,cnx,ax,logo=logo,labels=labels)
    plt.axis('off'); plt.axis('equal')
    plt.tight_layout(); plt.show()


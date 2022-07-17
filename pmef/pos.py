from numpy import zeros, array, average, append, delete, insert, arange, hstack
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
import matplotlib.tri as tri
import matplotlib.cm


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

def stress(u, Mesh, ElementData, ProblemData, ModelData):
    "Función que obtiene esfuerzos en los nodos."
    from pmef.pro import ShapeFunction, DofMap

    n, dim = ElementData.nodes, ProblemData.SpaceDim

    if dim == 1:
        if n == 1:
            print("Debes programar para %s puntos aún." %n)
        elif n == 2:
            print("Debes programar para %s puntos aún." %n)
        elif n == 4:
            print("Debes programar para %s puntos aún." %n)
        else:
            print("Debes programar para %s puntos aún."%n)
            
    elif dim == 2:
        E,v = ModelData.E, ModelData.v # Estado plano de esfuerzos
        if ModelData.state == 'PlaneStress': pass
        elif ModelData.state == 'PlaneStrain': E,v = E/(1.0-v*2),v/(1-v)
        else: print('El estado plano solo puede ser "PlaneStress" o "PlaneStrain"')
        
        # Formando Matriz D
        D = zeros((3,3))
        D[0,0], D[1,1], D[0,1], D[1,0]= 1.0, 1.0, v, v
        D[2,2] = 0.5*(1.0-v)
        D = E*D/(1-v**2)

        if n == 3:
            gp     = zeros((3,3))
            a = 1
            gp[1,0], gp[1,1], gp[1,2] = a, 0, 0
            gp[2,0], gp[2,1], gp[2,2] = 0, a, 0
        elif n == 4: # 4 puntos para Quad4
            gp     = zeros((3,4))
            a = 1
            gp[1,0], gp[1,1], gp[1,2], gp[1,3] = -a, a, a,-a
            gp[2,0], gp[2,1], gp[2,2], gp[2,3] = -a,-a, a, a
        else:
            print("Debes programar para %s puntos aún"%n)
        
        N = Mesh.NN
        sig = zeros((N,3),'float')
        count = zeros(Mesh.NN,'int')
        m = ElementData.dof
        for connect_element in Mesh.Conex:
            points = gp.T
            points[:,0] = connect_element
            x_element = Mesh.Nodos[connect_element]

            for point in points:
                nodo = int(point[0])
                [N, dN,ddN, j] = ShapeFunction(x_element, point, ElementData.type)
                B = zeros((3,m*n))
                for i in range(m):
                    B[i, i::m] = dN[i]
                B[2, 0::m] = dN[1]
                B[2, 1::m] = dN[0]
                dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
                sig[nodo] = sig[nodo] + D@B@u[dof]
            count[connect_element] += 1
        for i in range(Mesh.NN):
            sig[i] = sig[i]/count[i]

    elif dim == 3:
        E,v = ModelData.E, ModelData.v
        D = zeros((6, 6))
        λ = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
        μ = E / (2.0 * (1.0 + v))
        D[0, 0] = D[1, 1] = D[2, 2] = λ + 2*μ 
        D[3, 3] = D[4, 4] = D[5, 5] = μ
        D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = λ

        if n == 8:
            a = 1.0
            gp = zeros((4, 8))
            gp[1, 0], gp[1, 1], gp[1, 2], gp[1, 3], gp[1, 4], gp[1, 5], gp[1, 6], gp[1, 7] = -a, a, a, -a, -a, a, a, -a
            gp[2, 0], gp[2, 1], gp[2, 2], gp[2, 3], gp[2, 4], gp[2, 5], gp[2, 6], gp[2, 7] = -a, -a, a, a, -a, -a, a, a
            gp[3, 0], gp[3, 1], gp[3, 2], gp[3, 3], gp[3, 4], gp[3, 5], gp[3, 6], gp[3, 7] = -a, -a, -a, -a, a, a, a, a

        else:
            print("Debes programar para %s puntos aún." %n)
        N = Mesh.NN
        sig = zeros((N,6),'float')
        count = zeros(Mesh.NN,'int')
        m = ElementData.dof

        for connect_element in Mesh.Conex:
            points = gp.T
            points[:,0] = connect_element
            x_element = Mesh.Nodos[connect_element]
            for point in points:
                nodo = int(point[0])

                [N, dN,ddN, j] = ShapeFunction(x_element, point, ElementData.type)
                B = zeros((6,m*n))
                for i in range(m):
                    B[i, i::m] = dN[i]
                B[3, 0::m] = dN[1]
                B[3, 1::m] = dN[0]
                B[4, 1::m] = dN[2]
                B[4, 2::m] = dN[1]
                B[5, 0::m] = dN[2]
                B[5, 2::m] = dN[0]
                dof = DofMap(ElementData.dof, connect_element, ElementData.nodes)
                sig[nodo] = sig[nodo] + D@B@u[dof]

            count[connect_element] += 1
            
        for i in range(Mesh.NN):
            sig[i] = sig[i]/count[i]

        vmises = zeros((Mesh.NN,1),'float')
        for i in range(Mesh.NN):
            vmises[i] = (((sig[i,0]-sig[i,1])**2 + (sig[i,1]-sig[i,2])**2 + (sig[i,0]-sig[i,2])**2 + 6*(sig[i,3]**2 + sig[i,4]**2 + sig[i,5]**2))/2)**0.5
        sig = hstack((sig,vmises))
    else:
        print("Debes programar para %sD aún."%dim)

    return sig

def graph(x,cnx,ax,color='k',d=0.01,logo=True,labels=False):
    '''
    Función que grafica los nodos y elementos del modelo.
    '''
    if logo == True:
        xrng, yrng = plt.xlim(),plt.ylim()
        L = min(xrng[1]-xrng[0],yrng[1]-yrng[0])
        im = image.imread('https://jpi-ingenieria.com/images/logoJPI.png')
        ax.imshow(im,aspect='auto',extent=(xrng[0], xrng[0]+0.2*L, yrng[0], yrng[0]+0.2*L), zorder=10,alpha=0.7)

    if len(x.shape)==1:
        dim = 1
        NN = len(x)
    else:
        (NN,dim) = x.shape

    if dim == 1:
        for ii in range(len(x)):
            if labels == True:
                ax.plot(x[ii],0.0,color+'o',markersize=1.5)
                ax.annotate('%i'%(ii),(x[ii],0.0),
                                xytext=(x[ii]+d,0.0+d),color='black')
        ie=0
        for e in cnx:
            xx = x[e]
            ax.plot(xx,[0.0,0.0],color,lw = 1.0,alpha=1.0)
            if labels == True:
                xa,ya=average(xx),0.0
                ax.annotate('%i'%(ie),(xa,ya+d),color='blue',fontsize=8.)
            ie += 1
        Lx = max(x) - min(x)
        plt.axis([average(x)-1.02*Lx/2,average(x)+1.02*Lx/2,-Lx/4,Lx/4])

    elif dim == 2:
        for ii in range(len(x)):
            if labels == True:
                ax.plot(x[ii,0],x[ii,1],'ko',markersize=1.5)
                ax.annotate('%i'%(ii),(x[ii,0],x[ii,1]),
                                xytext=(x[ii,0]+d,x[ii,1]+d),color='black')
        ie=0
        for e in cnx:
            xx = x[append(e,e[0])]
            ax.plot(xx[:,0],xx[:,1],'k',lw = 0.3,alpha=0.5)
            if labels == True:
                xa,ya=average(x[e,0]),average(x[e,1])
                ax.annotate('%i'%(ie),(xa,ya),color='blue',fontsize=8.)
            ie += 1

    elif dim == 3:
        for ii in range(len(x)):
            if labels == True:
                ax.plot(x[ii,0],x[ii,1],x[ii,2],color+'o',markersize=1.5)
                ax.text(x[ii,0],x[ii,1],x[ii,2],'%i'%(ii),color='black')
        ie=0
        for e in cnx:
            xx = x[[e[0],e[1],e[2],e[3],e[0],e[4],e[5],e[6],e[7],e[4]]]
            ax.plot(xx[:,0],xx[:,1],xx[:,2],color,lw = 0.3,alpha=0.5)
            ax.plot(x[[e[1],e[5]],0],x[[e[1],e[5]],1],x[[e[1],e[5]],2],'k',lw = 0.3,alpha=0.5)
            ax.plot(x[[e[2],e[6]],0],x[[e[2],e[6]],1],x[[e[2],e[6]],2],'k',lw = 0.3,alpha=0.5)
            ax.plot(x[[e[3],e[7]],0],x[[e[3],e[7]],1],x[[e[3],e[7]],2],'k',lw = 0.3,alpha=0.5)
            if labels == True:
                xa,ya,za = average(x[e,0]),average(x[e,1]),average(x[e,2])
                ax.text(xa,ya,za,'%i'%(ie),color='blue',fontsize=8.)
            ie += 1
        L = x.max()-x.min()
        ax.plot(average(x[:,0])+L/2,average(x[:,1])+L/2,average(x[:,2])+L/2,alpha=0.0)
        ax.plot(average(x[:,0])-L/2,average(x[:,1])-L/2,average(x[:,2])-L/2,alpha=0.0)
        logo = False

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

def plot2D_deform(up,defo,cnx,ax,color='RdYlGn_r',bar_label='Resultados',logo=True,labels=False,elems=True):
    '''
    Función que realiza el ploteo de nodos, elementos y resultados.
    '''
    if cnx.shape[1]==3: 
        triangulation = tri.Triangulation(defo[:,0], defo[:,1], triangles=cnx)
    elif cnx.shape[1]==4:
        tris = quads_to_tris(cnx)
        triangulation = tri.Triangulation(defo[:,0], defo[:,1], triangles=tris)
    else: print('pmef no soporta el mesh proporcionado para la gráfica.')
    ax.tricontourf(triangulation, up, cmap=color, alpha=1.0,levels=50)
    norm = colors.Normalize(vmin=min(up), vmax=max(up))
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color),
                    orientation='vertical',label=bar_label)
    if elems==True:
        graph(defo,cnx,ax,logo=logo,labels=labels)
    plt.axis('off'); plt.axis('equal')
    plt.tight_layout()


def V_insert(V, BC_data, ElementData):
    '''Función que agrega valores nulos aun vector en pocisiones especificadas
    '''
    for BC in BC_data:
      if int(BC[1])==1:
        i = int(BC[0])*ElementData.dof + int(BC[2])-1
        V = insert(V, i, 0)
    return V

def K_reduce(K, BC_data, ElementData):
    '''Función que elimina grados de libertad que no se desea analizar
    '''
    (n,m) = K.shape
    dof = arange(0,n,1,'int')
    li = []
    for BC in BC_data:
      if int(BC[1])==1:
        li.append(int(BC[0])*ElementData.dof + int(BC[2])-1)
    dof = delete(dof,li)
    Kr = K[dof][:,dof]
    return Kr
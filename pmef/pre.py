from numpy import zeros, array, dot, delete, append, unique
from numpy import sin, arccos, arctan2
from numpy.linalg import norm


def LinearMesh(L,Ne,x0=0):
    '''
    Función que retorna nodos y conexiones para elementos finitos en 1D.
    Donde:
            L:      Longitud total de la barra.
            Ne:     Número de elementos a crear.
            x0:     Posición del primer nodo.
    '''
    L_element=L/Ne
    # Crea arreglo de nodos
    nodos = zeros(Ne+1)
    for i in range(Ne+1): 
        nodos[i]=x0 + i*L_element
    # Crea arreglo de conexiones
    conex = zeros((Ne,2),'int')
    conex[:,0],conex[:,1] = range(Ne), range(1,Ne+1)
    class Mesh:
        NN = len(nodos)
        Nodos = nodos
        NC = len(conex)
        Conex = conex
    return Mesh

def BC_2Dx(X,dy,x,tipo,gdl,val):
    ''' Función que busca el nodo donde se aplicará las CB, especificando
    una distancia Y y los tramos en x. (Utiliza un radio de búsqueda)
    Input:
            dy:			Distancia Y donde se aplicaran CB
            x:          intervalos de X donde se aplicaran CB [x1,x2]
            X:          Matriz que contiene las coordenadas de los nodos.
    Output:
            BC_data:    Matriz que define las condiciones de borde (la 
                        columnas corresponden a: Nodo, Tipo, GDL y Valor)
    '''
    NN,k=len(X),0
    BC_data = zeros((10000,4),dtype='float32')
    x1,x2 = x[0],x[1]
    k= 0
    for i in range(NN):
        if abs(X[i,1]-dy)<=10e-6:
            if X[i,0]>=x1 and X[i,0]<=x2:
                for g in gdl:
                    BC_data[k] = [i,tipo,g,val]
                    # print('Applying BC:',i,tipo,g,val)
                    k = k + 1
            else: continue
        else: continue
    BC_data = BC_data[:k]
    print('Se crearon %i condiciones de Borde.'%len(BC_data))
    return BC_data

def BC_2Dy(X,dx,y,tipo,gdl,val):
    ''' Función que busca el nodo donde se aplicará las CB, especificando
    una distancia Y y los tramos en x. (Utiliza un radio de búsqueda)
    Input:
            dx:			Distancia X donde se aplicaran CB
            y:          intervalos de Y donde se aplicaran CB [y1,y2]
            X:          Matriz que contiene las coordenadas de los nodos.
    Output:
            BC_data:    Matriz que define las condiciones de borde (la 
                        columnas corresponden a: Nodo, Tipo, GDL y Valor)
    '''
    NN,k=len(X),0
    BC_data = zeros((10000,4),dtype='float32')
    y1,y2 = y[0],y[1]
    k= 0
    for i in range(NN):
        if abs(X[i,0]-dx)<=10e-6:
            if X[i,1]>=y1 and X[i,1]<=y2:
                for g in gdl:
                    BC_data[k] = [i,tipo,g,val]
                    k = k + 1
            else: continue
        else: continue
    BC_data = BC_data[:k]
    print('Se crearon %i condiciones de Borde.'%len(BC_data))
    return BC_data



# Funciones para delaunay triangulation
def circumcenter(tri):
    s = zeros(3)
    pts = [2,0,1,2,0]
    for i in range(1,4):
        r1 = tri[pts[i-1]]-tri[pts[i]]
        r2 = tri[pts[i+1]]-tri[pts[i]]
        s[i-1]=sin(2*arccos(dot(r1,r2)/(norm(r1)*norm(r2))))
    xo = dot(tri[:,0],s)/sum(s)
    yo = dot(tri[:,1],s)/sum(s)
    r = sum((tri[2]-[xo,yo])**2)**0.5
    return xo,yo,r

def getPolygon(xyo,elems,xd):
    pol = elems[0]
    for e in elems[1:]:
        pol = append(pol,e)
    pol = unique(pol)
    ang = zeros(len(pol))
    for i in range(len(pol)):
        x1,y1 = xd[pol[i]]
        ang[i]=arctan2(y1-xyo[1],x1-xyo[0])
    args = ang.argsort()
    pol = pol[args]
    return append(pol,pol[0])

def updateMesh(xyo,xd,elems,selec):
    local_elems = elems[selec]
    if len(local_elems)==0: print('*'*20+'No triangle found!'+'*'*20)
    else: pol = getPolygon(xyo,local_elems,xd)
    elems2 =delete(elems, selec, axis=0)
    xd = append(xd,[xyo],axis=0)
    for i in range(len(pol)-1):
        temp = [len(xd)-1,pol[i],pol[i+1]]
        elems2 = append(elems2,[temp],axis=0)
    return xd,elems2

def delaunay(coor):
    maxX,maxY = max(coor[:,0]),max(coor[:,1])
    minX,minY = min(coor[:,0]),min(coor[:,1])

    xi = minX - 0.1*(maxX-minX)
    yi = minY - 0.1*(maxY-minY)
    xf = maxX + 0.1*(maxX-minX)
    yf = maxY + 0.1*(maxY-minY)

    x = array([[xi,yi],[xf,yi],[xf,yf],[xi,yf]])
    cnx = array([[0,1,2],[0,2,3]])

    for i in range(len(coor)):
        # print('node %i: (%8.5f,%8.5f)'%(i,coor[i,0],coor[i,1]))
        ie = 0
        selec =[]
        for e in cnx:
            xe = x[e]
            xo,yo,r = circumcenter(xe)
            d = sum((coor[i]-[xo,yo])**2)**0.5
            if d <= 1.000001*r: 
                selec.append(ie)
                # print('The point %i is inside circle of element %i :D'%(i,ie))
            else:
                pass
                # print('The point %i is out of circle of element %i :('%(i,ie))
            ie += 1
        x,cnx = updateMesh(coor[i],x,cnx,selec)
    x = coor
    selec = []
    for ie in range(len(cnx)):
        for n in cnx[ie]:
            if n<=3:
                selec.append(ie)
                break
    cnx =delete(cnx, selec, axis=0)-4
    return cnx


# Funcion que genera nodos para analizar cimentaciones

def founMesh(x1,y1,x2,y2,mz1):
    coor = zeros((10000,2))
    ni, nj = int(2*x2/mz1), int(y2/mz1)
    k = 0
    for j in range(nj+1):
        for i in range(ni+1):
            if (mz1*i-x2)>=-x1 and (mz1*i-x2)<=x1 and -mz1*j>=-y1:
                continue
            coor[k] = [mz1*i-x2,-mz1*j]
            k = k + 1 
    coor = coor[:k]
    return coor



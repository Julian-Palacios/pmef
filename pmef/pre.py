from numpy import zeros

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
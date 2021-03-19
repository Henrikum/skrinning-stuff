# Geometry utilities
# Created 2021-03-19


def area(geom='hex', dims=[]):
    if geom == 'rect':
        a = dims[0]*dims[1]
    else:  # hex
        r = dims[0]/2
        h = 3**0.5/2*r
        a = 3*h*r
    return a


def periphery(geom='hex', dims=[]):
    if geom == 'rect':
        p = 2*(dims[0] + dims[1])
    else:  # hex
        r = dims[0]/2
        p = 6*r
    return p


def charLength(geom='hex', dims=[]):
    a = area(geom, dims)
    p = periphery(geom, dims)
    return 4*a/p
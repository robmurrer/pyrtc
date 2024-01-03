
import math

EPSILON = 1e-6

tuple3 = tuple[float,float,float] # colors
tuple4 = tuple[float,float,float,float] #vector, point
matrix = list[list[float]]

def get_vector(x:float,y:float,z:float)->tuple4:
    return (x,y,z,0)

def get_point(x:float,y:float,z:float)->tuple4:
    return (x,y,z,1)

def get_matrix(width: int, height: int)->matrix:
    return [[0 for _ in range(height)] for _ in range(width)]


# this can be replaced with math.isclose()
def float_is_equal(a:float, b:float, eps=EPSILON)->bool:
    return math.fabs(a-b) < eps

def tuple_is_point(tuple:tuple4)->bool:
    return int(tuple[-1]) == 1 #do we need this cast? does it slow us down?

def tuple_is_equal(a:tuple4, b:tuple4, eps=EPSILON)->bool:
    return \
        float_is_equal(a[0], b[0], eps) and \
        float_is_equal(a[1], b[1], eps) and \
        float_is_equal(a[2], b[2], eps) and \
        float_is_equal(a[3], b[3], eps)

def tuple3_equal(a:tuple3, b:tuple3, eps=EPSILON):
    return \
        float_is_equal(a[0], b[0], eps) and \
        float_is_equal(a[1], b[1], eps) and \
        float_is_equal(a[2], b[2], eps)

def tuple_neg(a:tuple4)->tuple4:
    return (-a[0], -a[1], -a[2], -a[3])

def tuple_add(a:tuple4,b:tuple4)->tuple4:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3])

def tuple_sub(a:tuple4,b:tuple4)->tuple4:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3])

def tuple_mul_scale(tuple:tuple4, scalar:float)->tuple4:
    return (scalar*tuple[0], scalar*tuple[1], scalar*tuple[2], scalar*tuple[3])

def tuple_div_scale(tuple:tuple4, scalar:float)->tuple4:
    return tuple_mul_scale(tuple,1/scalar)

def tuple_mag(tuple:tuple4)->float:
    return math.sqrt(tuple[0]**2 + tuple[1]**2 + tuple[2]**2 + tuple[3]**2)

def tuple_norm(tuple:tuple4)->tuple4:
    mag = tuple_mag(tuple)
    return tuple_div_scale(tuple, mag)

def tuple_dot(a:tuple4, b:tuple4)->float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

X,Y,Z = range(3) 
def tuple_cross(a:tuple3, b:tuple3)->tuple4:
    return (a[Y]*b[Z]-a[Z]*b[Y], 
            a[Z]*b[X]-a[X]*b[Z],
            a[X]*b[Y]-a[Y]*b[X],0) #don't forget tuples are 4 but we ignore last in cross

R,G,B = range(3)
def get_color(r:float, g:float, b:float)->tuple:
    return (r,g,b)

def color_add(a:tuple3, b:tuple3):
    return (a[R]+b[R],a[G]+b[G],a[B]+b[B])

def color_sub(a:tuple3, b:tuple3):
    return (a[R]-b[R],a[G]-b[G],a[B]-b[B])

def color_scale(a:tuple3, scale:float):
    return (a[R]*scale,a[G]*scale,a[B]*scale)

def color_mul(a:tuple3, b:tuple3):
    return (a[R]*b[R],a[G]*b[G],a[B]*b[B])

def matrix_is_equal(m1: matrix, m2: matrix)->bool:
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            if not math.isclose(m1[i][j], m2[i][j]):
                return False
    return True

def matrix_mul(m1: matrix, m2: matrix)->matrix:
    result = [[0 for _ in range(len(m2[0]))] for _ in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                result[i][j] += m1[i][k] * m2[k][j]
    return result


def matrix_mul_tuple(matrix: matrix, t: tuple4) -> tuple4:
    result = [0, 0, 0, 0]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i] += matrix[i][j] * t[j]
    return tuple(result)

def matrix_transpose(matrix: matrix)->matrix:
    result = get_matrix(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[j][i] = matrix[i][j]
    return result

def matrix_det(matrix: matrix)->float:
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

def matrix_submatrix(matrix: matrix, row: int, col: int)->matrix:
    result = get_matrix(len(matrix)-1, len(matrix[0])-1)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i != row and j != col:
                result[i if i < row else i-1][j if j < col else j-1] = matrix[i][j]
    return result

def test_matrix():
    m1 = get_matrix(4,4)
    m1[0] = [1,2,3,4]
    m1[1] = [5.5,6.5,7.5,8.5]
    m1[2] = [9,10,11,12]
    m1[3] = [13.5,14.5,15.5,16.5]

    assert(float_is_equal(m1[0][0], 1))
    assert(float_is_equal(m1[0][3], 4))
    assert(float_is_equal(m1[1][0], 5.5))
    assert(float_is_equal(m1[1][2], 7.5))
    assert(float_is_equal(m1[2][2], 11))
    assert(float_is_equal(m1[3][0], 13.5))
    assert(float_is_equal(m1[3][2], 15.5))

    m2 = get_matrix(2,2)
    m2[0] = [-3,5]
    m2[1] = [1,-2]
    assert(float_is_equal(m2[0][0], -3))
    assert(float_is_equal(m2[0][1], 5))
    assert(float_is_equal(m2[1][0], 1))
    assert(float_is_equal(m2[1][1], -2))

    m1 = get_matrix(4,4)
    m1[0] = [1,2,3,4]
    m1[1] = [5,6,7,8]
    m1[2] = [9,8,7,6]
    m1[3] = [5,4,3,2]

    m2 = get_matrix(4,4)
    m2[0] = [1,2,3,4]
    m2[1] = [5,6,7,8]
    m2[2] = [9,8,7,6]
    m2[3] = [5,4,3,2]
    assert(matrix_is_equal(m1,m2))

    m3 = get_matrix(4,4)
    m3[0] = [2,3,4,5]
    m3[1] = [6,7,8,9]
    m3[2] = [8,7,6,5]
    m3[3] = [4,3,2,1]
    assert(not matrix_is_equal(m1,m3))

    m1 = get_matrix(4,4)
    m1[0] = [-2, 1, 2, 3]
    m1[1] = [3, 2, 1, -1]
    m1[2] = [4, 3, 6, 5]
    m1[3] = [1, 2, 7, 8]

    mul = matrix_mul(m2,m1)

    m3 = get_matrix(4,4)
    m3[0] = [20, 22, 50, 48]
    m3[1] = [44, 54, 114, 108]
    m3[2] = [40, 58, 110, 102]
    m3[3] = [16, 26, 46, 42]
    assert(matrix_is_equal(mul, m3))

    m1 = get_matrix(4,4)
    m1[0] = [1,2,3,4]
    m1[1] = [2,4,4,2]
    m1[2] = [8,6,4,1]
    m1[3] = [0,0,0,1]

    t1 = (1, 2, 3, 1)

    m2 = matrix_mul_tuple(m1, t1)
    assert(tuple_is_equal(m2, (18, 24, 33, 1)))

    id4x4 = get_matrix(4,4)
    id4x4[0] = [1,0,0,0]
    id4x4[1] = [0,1,0,0]
    id4x4[2] = [0,0,1,0]
    id4x4[3] = [0,0,0,1]

    m1 = get_matrix(4,4)
    m1[0] = [0,1,2,4]
    m1[1] = [1,2,4,8]
    m1[2] = [2,4,8,16]
    m1[3] = [4,8,16,32]
    assert(matrix_is_equal(matrix_mul(id4x4, m1), m1))
    assert(matrix_is_equal(matrix_mul(m1, id4x4), m1))

    m1 = get_matrix(4,4)
    m1[0] = [0,9,3,0]
    m1[1] = [9,8,0,8]
    m1[2] = [1,8,5,3]
    m1[3] = [0,0,5,8]

    m2 = get_matrix(4,4)
    m2[0] = [0,9,1,0]
    m2[1] = [9,8,8,0]
    m2[2] = [3,0,5,5]
    m2[3] = [0,8,3,8]

    m3 = matrix_transpose(m1)
    assert(matrix_is_equal(m3, m2))
    assert(matrix_is_equal(matrix_transpose(id4x4), id4x4))

    m2x2 = get_matrix(2,2)
    m2x2[0] = [1,5]
    m2x2[1] = [-3,2]    

    det = matrix_det(m2x2)
    assert(float_is_equal(det, 17))

    m3x3 = get_matrix(3,3)
    m3x3[0] = [1,5,0]
    m3x3[1] = [-3,2,7]
    m3x3[2] = [0,6,-3]

    m2x2 = get_matrix(2,2)
    m2x2[0] = [-3,2]
    m2x2[1] = [0,6]
    assert(matrix_is_equal(matrix_submatrix(m3x3, 0, 2), m2x2))

    m4x4 = get_matrix(4,4)
    m4x4[0] = [-6,1,1,6]
    m4x4[1] = [-8,5,8,6]
    m4x4[2] = [-1,0,8,2]
    m4x4[3] = [-7,1,-1,1]

    m3x3 = get_matrix(3,3)
    m3x3[0] = [-6,1,6]
    m3x3[1] = [-8,8,6]
    m3x3[2] = [-7,-1,1]
    assert(matrix_is_equal(matrix_submatrix(m4x4, 2, 1), m3x3))

    print("Matrix Tests Passed")
test_matrix()

def test_colors():
    c1 = (0.9,0.6,0.75)
    c2 = (0.7,0.1,0.25)
    assert(tuple3_equal(color_add(c1,c2), (1.6,0.7,1.0)))
    assert(tuple3_equal(color_sub(c1,c2), (0.2,0.5,0.5)))
    assert(tuple3_equal(color_scale((0.2,0.3,0.4), scale=2), (0.4,0.6,0.8)))
    assert(tuple3_equal(color_mul((1,0.2,0.4), (0.9,1,0.1)),(0.9,0.2,0.04)))

    print("Color Tests Passed")
test_colors()

def test_tuples():
    t1_point = (4.3,-4.2,3.1,1.0)
    t2_vector = (4.3,-4.2,3.1,0.0)

    assert(tuple_is_point(t1_point))
    assert(not tuple_is_point(t2_vector))
    assert(tuple_is_equal(t1_point,t1_point))
    assert(not tuple_is_equal(t1_point,t2_vector))

    a1 = (3,-2,5,1)
    a2 = (-2,3,1,0)
    assert(tuple_is_equal(tuple_add(a1,a2), (1,1,6,1)))

    p1 = (3,2,1,1)
    p2 = (5,6,7,1)
    p1_sub_p2 = tuple_sub(p1,p2)
    assert(not tuple_is_point(p1_sub_p2))
    assert(tuple_is_equal(p1_sub_p2, (-2,-4,-6,0)))

    v1 = get_vector(3,2,1)
    v2 = get_vector(5,6,7)
    v1_sub_v2 = tuple_sub(v1,v2)
    assert(tuple_is_equal(v1_sub_v2, get_vector(-2,-4,-6)))

    a = get_vector(1,2,-3)
    a_neg = tuple_neg(a)
    assert(tuple_is_equal(a_neg, get_vector(-1,-2,3)))

    p2 = tuple_mul_scale((1,-2,3,-4), 3.5)
    assert(tuple_is_equal(p2, (3.5, -7, 10.5, -14)))

    p3 = tuple_mul_scale((1,-2,3,-4), 0.5)
    assert(tuple_is_equal(p3, (0.5, -1, 1.5, -2)))

    p4 = tuple_div_scale((1,-2,3,-4), 2)
    assert(tuple_is_equal(p4, (0.5, -1, 1.5, -2)))

    m1 = tuple_mag(get_vector(1, 0, 0))
    assert (float_is_equal(m1, 1))

    m2 = tuple_mag(get_vector(0, 1, 0))
    assert (float_is_equal(m1, 1))

    m3 = tuple_mag(get_vector(0, 0, 1))
    assert (float_is_equal(m1, 1))

    m4 = tuple_mag(get_vector(1, 2, 3))
    assert (float_is_equal(m4, math.sqrt(14)))

    m5 = tuple_mag(get_vector(-1, -2, -3))
    assert (float_is_equal(m5, math.sqrt(14)))

    n1 = tuple_norm((get_vector(4,0,0)))
    assert (tuple_is_equal(n1, (1,0,0,0)))

    assert(float_is_equal(1.23, 1.231, eps=1e-2))
    assert(not float_is_equal(1.23, 1.231, eps=1e-3))

    n2 = tuple_norm((get_vector(1,2,3)))
    n3 = get_vector(0.26726, 0.53452, 0.80178)
    assert(tuple_is_equal(n2, n3, eps=1e-3))

    d1 = tuple_dot(get_vector(1,2,3), get_vector(2,3,4))
    assert(float_is_equal(d1, 20))

    c1 = tuple_cross(get_vector(1,2,3), get_vector(2,3,4))
    assert(tuple_is_equal(c1, get_vector(-1,2,-1)))

    c2 = tuple_cross(get_vector(2,3,4), get_vector(1,2,3))
    assert(tuple_is_equal(c2, get_vector(1,-2,1)))

    print("Tuple Tests Passed")
test_tuples()
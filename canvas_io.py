import prim1 as pv
import numpy as np

canvas = list[list[list]]

def get_canvas_np(width: int, height: int)->np.ndarray:
    return np.zeros((width,height,3), dtype=np.double)

def get_canvas(width: int, height: int)->canvas:
    return [[[0 for i in range(3)] for j in range(width)] for k in range(height)]

CAP_COLOR = 255
def canvas_write_pixel(canvas: canvas, x: int, y: int, color: pv.tuple3):
    assert(x < len(canvas))
    assert(y < len(canvas[0]))
    assert(x >=0 and y >=0)

    canvas[x][y] = color

    #canvas[x][y][0] = color[0]
    #canvas[x][y][1] = color[1]
    #canvas[x][y][2] = color[2]


def canvas_to_ppm_str(canvas: canvas)->str:
    width = len(canvas)
    assert(width >=0)
    height = len(canvas[0])
    assert(height >=0)
    colors = len(canvas[0][0])
    assert(colors == 3)


def canvas_np_to_ppm_str(canvas: np.ndarray)->str:
    pass
 
def test_Canvas():
    c = get_canvas(100,100)
    canvas_write_pixel(c, 50, 50, (1,2,3))
    assert(pv.float_is_equal(c[50][50][0], 1))
    assert(pv.float_is_equal(c[50][50][2], 3))

    canvas_write_pixel(c, 65, 75, (1,2,3))
    assert(pv.float_is_equal(c[65][75][2], 3))

    print("Canvas Tests Pass")

test_Canvas()
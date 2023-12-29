import prim1 as pv
import numpy as np

canvas = list[list[list]]

def get_canvas_np(width: int, height: int)->np.ndarray:
    return np.zeros((width,height,3), dtype=np.double)

def get_canvas(width: int, height: int)->canvas:
    return [[[0 for i in range(3)] for j in range(height)] for k in range(width)]

def canvas_write_pixel(canvas: canvas, x: int, y: int, color: pv.tuple3):
    assert(y < len(canvas[0]))
    assert(x < len(canvas))
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

    # really slow/weird without numpy?
    c = np.array(canvas).reshape(-1)

    # need to clamp to 0 to 255
    max = np.max(c)
    min = np.min(c)
    #assert(min >= 0)
    CAP_COLOR = 255
    c = np.clip(c, 0, CAP_COLOR)
    c = CAP_COLOR*(c / max)
    
    return c

def canvas_np_to_ppm_str(canvas: np.ndarray)->str:
    pass
 
def test_Canvas():
    c = get_canvas(100,100)
    canvas_write_pixel(c, 50, 50, (1,2,3))
    assert(pv.float_is_equal(c[50][50][0], 1))
    assert(pv.float_is_equal(c[50][50][2], 3))

    canvas_write_pixel(c, 65, 75, (1,2,3))
    assert(pv.float_is_equal(c[65][75][2], 3))

    c = get_canvas(5,5)
    canvas_write_pixel(c, 1, 1, (1,2,3))
    canvas_write_pixel(c, 0, 0, (1,1,1))
    cstr = canvas_to_ppm_str(c)

    c = get_canvas(5,3)
    canvas_write_pixel(c, 0, 0, (1.5,0,0))
    canvas_write_pixel(c, 2, 1, (0, 0.5, 0))
    canvas_write_pixel(c, 4, 2, (-0.5, 0, 1))
    cstr = canvas_to_ppm_str(c)
    print(cstr)

    print("Canvas Tests Pass")
test_Canvas()
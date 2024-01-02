import prim1 as pr
import canvas_io as cvi

ball_pos = pr.get_point(1,1,0)
#ball_vel = pr.tuple_norm(pr.tuple_mul_scale(pr.get_vector(1,1,0), 1.25))
ball_vel = pr.get_vector(15,15,0)

env_gravity = pr.get_vector(0,-1,0)
#env_wind = pr.get_vector(-0.01,0,0)
env_wind = pr.get_vector(0,0,0)

def tick(ball_pos, ball_vel, env_gravity, env_wind):
    return (pr.tuple_add(ball_pos,ball_vel), pr.tuple_add(pr.tuple_add(ball_vel,env_gravity),env_wind))


def sim(ball_pos, ball_vel, max_iters=1e3):
    pos_t = []
    for t in range(int(max_iters)):
        ball_pos, ball_vel = tick(ball_pos,ball_vel,env_gravity, env_wind)
        pos_t.append(ball_pos)
        if (ball_pos[1] <= 0): 
            print('ball hit ground')
            break

    return pos_t


pos = sim(ball_pos,ball_vel)
canvas = cvi.get_canvas(900,550)
for p in pos:
    x = int(p[0])
    y = int(550-p[1]) #swap y axis
    #y = int(p[1])
    cvi.canvas_write_pixel(canvas, x, y, (1,2,3))

#for i in range(550):
#    cvi.canvas_write_pixel(canvas, 10, i, (0,3,0))

#for i in range(900):
#    cvi.canvas_write_pixel(canvas, i, 10, (0,0,3))

#for i in range(900):
    #cvi.canvas_write_pixel(canvas, i, i, (3,3,3))

cvi.canvas_write_pixel(canvas, 10, 10, (3,0,0))
cvi.canvas_write_pixel(canvas, 899, 539, (3,0,0))

print("writing to disk")
import time
tic = time.perf_counter()
# cvi.canvas_to_ppm(canvas, "ball2.ppm")
toc = time.perf_counter()
#print("done: " + str(toc-tic) + " seconds")

import matplotlib.pyplot as plt
plt.imshow(canvas.swapaxes(0,1))
plt.show()
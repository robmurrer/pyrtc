import prim1 as pr
import canvas_io as cvi

ball_pos = pr.get_point(0,1,0)
ball_vel = pr.tuple_norm(pr.tuple_mul_scale(pr.get_vector(1,1.8,0), 11.25))

env_gravity = pr.get_vector(0,-0.1,0)
env_wind = pr.get_vector(-0.01,0,0)

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
    cvi.canvas_write_pixel(canvas, x, y, (1,2,3))

cvi.canvas_write_pixel(canvas, 0, 0, (10,0,0))
print("writing to disk")
import time
tic = time.perf_counter()
cvi.canvas_to_ppm(canvas, "ball2.ppm")
toc = time.perf_counter()
print("done: " + str(toc-tic) + " seconds")

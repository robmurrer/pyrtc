import prim1 as pr

ball_pos = pr.get_point(0,1,0)
ball_vel = pr.tuple_norm(pr.get_vector(1,1,0))

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


print(sim(ball_pos,ball_vel))
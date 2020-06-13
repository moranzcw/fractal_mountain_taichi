import taichi as ti
import numpy as np
import time
from PIL import Image

ti.init(arch=ti.gpu)

w, h = 800, 450
screen = ti.Vector(3, dt=ti.f32, shape=(w, h))

count_time = ti.var(ti.f32, shape=())

# 加载纹理
image = np.array(Image.open('texture.jpg'), dtype=np.float32)
image /= 256.0
tw, th = image.shape[0:2]
texture = ti.Vector(3, dt=ti.f32, shape=(tw, th))
texture.from_numpy(image)

count_time[None] = 0.0


# 双线性纹理过滤
@ti.func
def texture_bilinear_filter(u, v):
    u %= 1.0
    v %= 1.0
    u, v = u * tw, v * th
    left, right = int(u), int(u) + 1
    bottom, top = int(v), int(v) + 1
    t = u - left
    s = v - bottom
    col = ti.Vector([0.0, 0.0, 0.0])
    col = (1-t)*((1-s)*texture[left,bottom] + s*texture[right,bottom]) + \
            t*((1-s)*texture[left,top] + s*texture[right,top])
    return col


@ti.kernel
def draw():
    count_time += 0.1
    for i, j in screen:
        col = ti.Vector([0.0, 0.0, 0.0])
        p = ti.Vector([i / float(w), j / float(h), 1.0]) - 0.5
        d = ti.Vector([i / float(w), j / float(h), 1.0]) - 0.5
        p.z += count_time * 80.0
        d.y -= 0.4

        k = 1.5
        while k > 0.0:
            t = ti.Vector([0.0, 0.0, 0.0])
            e = ti.Vector([0.0, 0.0, 0.0])
            s = 0.5
            for m in range(0, 6):
                e = texture_bilinear_filter(0.3 + p.x * s / 3000.0,
                                            0.3 + p.z * s / 3000.0)
                s += s
                t += e / s
            col = 1.0 + d.x - t * k
            col.z -= 0.1
            if t.x > (p.y * 0.007 + 1.3):
                break
            p += d
            k -= 0.0015
        screen[i, j] = col


# video_manger = ti.VideoManager(output_dir='./results', framerate=10, automatic_build=False)
gui = ti.GUI("screen", (w, h))
for i in range(100000):
    draw()
    gui.set_image(screen.to_numpy())
    gui.show()
    # video_manger.write_frame(screen.to_numpy())

# video_manger.make_video(gif=True, mp4=True)

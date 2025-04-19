from PIL import Image, ImageDraw
import numpy as np
import time
import tkinter as tk

'''
Вращение:
вращение x y z=const и меняется проекция 
Камера остается на месте

'''

# Параметры камеры
# Тета считается от z при theta=90 камера ложится на плоскость xy
# фи считается от x.
theta = np.radians(45)  # угол наклона от горизонтали (вверх/вниз)
phi = np.radians(45)  # угол поворота в плоскости XY
r = 400  # расстояние от начала координат

# Положение камеры
x_cam = r * np.sin(theta) * np.cos(phi)
y_cam = r * np.sin(theta) * np.sin(phi)
z_cam = r * np.cos(theta)

# Вычисляю вектор направления координат (смотрит в 0 0 0)
forward = np.array([-x_cam, -y_cam, -z_cam])
forward /= np.linalg.norm(forward)  # Делаю его единичным

# смотри рисунок, там я пометил forward right и up
# forward- куда смотрим вычисляются векторными произведениями
# right-то что справа
# up -то что сверху
tmp = np.array([0, 0, 1]) if abs(forward[2]) < 0.999 else np.array(
    [0, 1, 0])  # tmp вектор, который не совпадет с forward
right = np.cross(tmp, forward)
right /= np.linalg.norm(right)
up = np.cross(forward, right)


# Определение функции (лента Мебиуса)
def parametric_function(u, v, alfa=2.0, beta=1.0):
    x = (alfa + v * np.cos(u / 2)) * np.cos(u)
    y = (alfa + v * np.cos(u / 2)) * np.sin(u)
    z = beta * v * np.sin(u / 2)
    return x, y, z


# Преобразуем точку в координаты относительно камеры
def rotate_to_camera(x, y, z):
    # Формируем матрицу поворота.
    rot_matrix = np.array([right, up, forward])
    vec = np.array([x - x_cam, y - y_cam, z - z_cam])
    x_new, y_new, z_new = rot_matrix @ vec
    return x_new, y_new, z_new


# Проекция 3D в 2D
def project_point(x, y, z, width, height, scale=50, d=1000):
    x, y, z = rotate_to_camera(x, y, z)
    if z <= 1e-3:
        return None
    factor = d / z
    x_proj = x * factor * scale + width // 2
    y_proj = -y * factor * scale + height // 2
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    point_size = min(max(0.35, int(0.2 / (distance / 100))), 5)
    if 0 <= x_proj < width and 0 <= y_proj < height:
        # вохвращаю координаты точки (измененные x и y, z прежняя)
        return int(x_proj), int(y_proj), z
    return None


def connect(p1, p2, draw, color="black"):
    # p1 и p2 — кортежи (x, y, z)
    x0, y0 = p1[0], p1[1]
    x1, y1 = p2[0], p2[1]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        draw.point((x0, y0), fill=color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x0 += sx
        if e2 < dx:
            error += dx
            y0 += sy

'''    
    1) ищу координаты пересечения
    2) сортирую координаты пересечения по возраст
    3) объединяю в пары 
    4) закршиваю между парами
'''
# vertices список вершин (x1, y1) (x2, y2)...
def draw_polygon(draw, vertices, fill_color, outline_color="black"):

    if not vertices:
        return

    #  границы полигона
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    min_y = min(ys)
    max_y = max(ys)

    for y in range(min_y, max_y + 1): #генерим горезонатльные линии (scanlines) вдоль всего полигона
        intersections = [] #пересечения скан-линии с ребрами
        n = len(vertices)
        for i in range(n):
            #пересекает ли скан-линия ребро?
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            if (v1[1] <= y and v2[1] > y) or (v2[1] <= y and v1[1] > y):
                dy = v2[1] - v1[1]
                if dy != 0:
                    t = (y - v1[1]) / dy
                    x_int = int(v1[0] + t * (v2[0] - v1[0])) #пересечение текущего ребра полигона со скан линией
                    intersections.append(x_int)

        # сортирую точки
        intersections.sort()

        # Рисуем горизонтальные отрезки между парами пересечений
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = intersections[i]
                x_end = intersections[i + 1]
                for x in range(x_start, x_end + 1):
                    draw.point((x, y), fill=fill_color)

    # Обводка полигона
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        connect(p1, p2, draw, color=outline_color)

#вращение в плоскости XY
def rotateXY(x, y, z, phi):
    phiR = np.radians(phi)
    x_rot=x*np.cos(phiR)-y*np.sin(phiR)
    y_rot=x*np.sin(phiR)+y*np.cos(phiR)
    z_rot=z
    return x_rot, y_rot, z_rot

#вращение в плоскости XZ
def rotateXZ(x, y, z, theta):
    thetaR = np.radians(theta)
    x_rot = x * np.cos(thetaR)+z * np.sin(thetaR)
    y_rot = y
    z_rot = -x*np.sin(thetaR)+z*np.cos(thetaR)
    return x_rot, y_rot, z_rot

width, height = 800, 800
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# Массивы параметров
u_range = np.linspace(0, 2 * np.pi, 30)
v_range = np.linspace(-0.5, 0.5, 15)

# Заполнение двумерного списка полигональных точек(по умолчанию там нихера)
poly_points = [[None] * len(v_range) for i in range(len(u_range))]

# создает кортежи (i, u) и (j, v)
def Paint(par):
    global img, draw  # Добавляем global для пересоздания изображения--------------------------------------------

    # Пересоздаем изображение при каждом обновлении
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            x, y, z = parametric_function(u, v)
            #поворот камеры
            if par==1:
                x_rot, y_rot, z_rot=rotateXY(x, y, z, 25) #--------------------------------сюда кнопку
            else:
                x_rot, y_rot, z_rot = rotateXZ(x, y, z, 25) #--------------------------------сюда кнопку

            projected = project_point(x_rot, y_rot, z_rot, width, height)
            poly_points[i][j] = projected

    # Генерация полигонов с глубиной
    polygons = []
    polygon_color = "lightblue"
    for i in range(len(u_range) - 1):
        for j in range(len(v_range) - 1):
            a, b, c, d = poly_points[i][j], poly_points[i][j + 1], poly_points[i + 1][j], poly_points[i + 1][j + 1]
            if a and b and c and d:
                # Вычисление средней глубины (зетки)
                depth = (a[2] + b[2] + c[2] + d[2]) / 4
                # Добавление полигона в список
                polygons.append((depth, (a, b, d, c), polygon_color))

    # Сортировка полигонов от дальнего к ближнему
    # сортирую по глубине и переворачиваю
    polygons.sort(key=lambda x: x[0], reverse=True)

    # Отрисовка полигонов через функцию draw_polygon
    for depth, points, color in polygons:
        # Приводим кортеж точек из проекций к списку координат (оставляем только x, y) т.е. перебираю по точкам абсд
        xy = [(p[0], p[1]) for p in points]
        draw_polygon(draw, xy, fill_color=color, outline_color="grey")


    for p in pointsx:
            draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="red")
    for p in pointsy:
            draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="green")
    for p in pointsz:
            draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="blue")

    # Обновляем изображение
    img.show()

# генерация осей x y и z
pointsx, pointsy, pointsz = [], [], []
for i in np.linspace(0, 10, 500):
    pz = project_point(0, 0, i, width, height)
    if pz:
        pointsz.append(pz)
    py = project_point(0, i, 0, width, height)
    if py:
        pointsy.append(py)
    px = project_point(i, 0, 0, width, height)
    if px:
        pointsx.append(px)

# отрисовка осей
for p in pointsx:
    draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="red")
for p in pointsy:
    draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="green")
for p in pointsz:
    draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="blue")

img.show()

#создание кнопок
root = tk.Tk()
btnXY = tk.Button(root, text="Вращать\nXY")
btnXY.config(command=lambda: Paint(1))
btnXY.pack(padx=120, pady=30)

btnXZ = tk.Button(root, text="Вращать\nXZ")
btnXZ.config(command=lambda: Paint(0))
btnXZ.pack(padx=120, pady=30)

root.title("Командный центр")
root.mainloop()

# Сохранение изображения
local_time = time.localtime()
year = local_time.tm_year
mounth = local_time.tm_mon
date = local_time.tm_mday
hour = local_time.tm_hour
minute = local_time.tm_min

name_file = "g " + str(minute) + "-" + str(hour) + "-" + str(date) + "-" + str(mounth) + "-" + str(year) + ".png"
img.save(name_file)

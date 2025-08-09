import turtle
from noise import pnoise1
import math
import random

turt = turtle.Turtle()
turt.speed(0)
turtle.bgcolor("black")
turt.color("white")
turt.width(2)

amplitude = 2
turn_strength = 0.1   # smaller turning
dt = 0.02
t = random.uniform(0, 1000)

x, y = 0, 0
angle = 0

turt.penup()
turt.goto(x, y)
turt.pendown()

screen = turtle.Screen()
w = screen.window_width()
h = screen.window_height()

for i in range(1000):
    turn_rate = 0
    for i in range(1, 6):
        turn_rate += pnoise1((t * i) + 100 * i) / i

    nudge = turn_rate * turn_strength

    angle += nudge
    x += math.cos(angle) * amplitude
    y += math.sin(angle) * amplitude

    turt.goto(x, y)
    t += dt


turtle.done()

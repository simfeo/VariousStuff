"""
Converter from uniform distribution to Gaus(normal) distribution
2 ways implemented:
- Ziggurat algorithm
- Box–Muller transform
"""

import math
import random
import plotly.express as px
import plotly.graph_objects as go

def clamp(i, m_min, m_max):
    if i < m_min:
        return m_min
    elif i > m_max:
        return m_max
    return i

def fi(x):
    return 1.0/math.sqrt(2*math.pi)*math.pow(math.e,-0.5*x*x)

max_y = fi(0)
def ziggurat(data):
    while True:
        indx = random.randrange(-500,500)
        x = indx / 100.0
        indx += 500
        y = random.uniform(0.0,max_y)
        y0 = fi(x)
        if y <= y0:
            data[indx] += 1
            break

def box_muller(data):
    fi = random.uniform(0.0,1)
    r = random.uniform(0.0,1)
    z0 = math.cos(2*math.pi*fi)*math.sqrt(-2*math.log2(r))
    z0_out = clamp(int(z0 * 100) + 500, 0, 999)
    data[z0_out] += 1

if __name__ == "__main__":
    # data = {"year": [1,2,3], "pop":[23,25,23]}
    data = {"x" : [x/100.0 for x in range (-500,500,1)], "y": []}
    data["y_z"] = [0] * len(data["x"])
    data["y"] = [0] * len(data["x"])

    generation_passes = 100000
    for i in range (100000):
        ziggurat(data["y_z"])
        box_muller(data["y"])
    
    data["y_norm"] = [fi(x/100 - 5)*generation_passes/100 for x in range(len(data["x"]))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode='lines', name="Box–Muller transform"))
    fig.add_trace(go.Scatter(x=data["x"], y=data["y_z"], mode='lines', name="Ziggurat"))
    fig.add_trace(go.Scatter(x=data["x"], y=data["y_norm"], mode='lines', name="Gauss"))
    fig.show()

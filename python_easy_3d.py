import math

def dot(vec1,vec2):
    return vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z

def magnitude(vec1):
    return math.sqrt(dot(vec1,vec1))

class Vector3:
    def __init__(self,x,y,z):
        self.x,self.y,self.z=x,y,z
        
    def normalize(self):
        mag = magnitude(self)
        return Vector3(self.x/mag,self.y/mag,self.z/mag)
        
    def __str__(self):
        return "x = {}, y ={}, z={}".format(self.x,self.y,self.z)

def project_on_plane(vec, normal):
    sqr_mag = magnitude(normal)
    if sqr_mag < 0.00001:
        return vec
    
    vdot = dot(vec, normal)
    res = Vector3(vec.x - normal.x*vdot/sqr_mag,
                 vec.y - normal.y*vdot/sqr_mag,
                 vec.z - normal.z*vdot/sqr_mag)
    return res
    
obj = Vector3(-0.1386645,0.6732482,0.7262982)
normal = Vector3(-0.9822585,0,-0.1875321).normalize()
print(project_on_plane(obj,normal))
plane = Vector3(0,0.8948231, -0.446421)
print(project_on_plane(plane,normal))

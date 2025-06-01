import sys
import imageio as iio
import queue
import numpy

sys.setrecursionlimit(10**4)

def load_image(image_name, step = 5):
    img = iio.v3.imread(image_name)

    len_y = len(img)
    len_x = len(img[0])

    empty_matrix = numpy.ndarray((len_y//step + 1, len_x//step + 1, 3), dtype=numpy.uint8)

    for y in range(0,len_y,step):
        for x in range(0,len_x,step):

            r,g,b = img[y][x]

            #resucing color variations
            r = r//30*30
            g = g//30*30
            b = b//30*30

            empty_matrix[y//step][x//step] = numpy.array([r,g,b],dtype=numpy.uint8)

    return empty_matrix


class Border:
    top = 0
    left = 0
    right = 0
    bottom = 0

    dimension_x = 0
    dimension_y = 0

    def __init__(self, y, x):
        self.right = x
        self.left = x
        self.top = y
        self.bottom = y

    def check(self, y, x):
        if x > self.right:
            self.right = x
            self.dimension_x = self.right - self.left
        if x < self.left:
            self.left = x
            self.dimension_x = self.right - self.left
        if y > self.bottom:
            self.bottom = y
            self.dimension_y = self.bottom - self.top
        if y < self.top:
            self.top = y
            self.dimension_y = self.bottom - self.top


def dfs(img,empty_matrix,counter,found_dict, y, x):
    r,g,b,a = img[y][x] 
    if a > 128 and empty_matrix[y][x] == 0:
        empty_matrix[y][x] = counter
        found_dict[counter].check(y,x)
        # print (y,x)
        dfs(img, empty_matrix, counter,found_dict, y + 1, x)
        dfs(img, empty_matrix, counter,found_dict, y - 1, x)
        dfs(img, empty_matrix, counter,found_dict, y , x + 1)
        dfs(img, empty_matrix, counter,found_dict, y , x - 1)

def find_all_objects(img, empty_matrix, counter, found_dict):
    len_y = len(img)
    len_x = len(img[0])
    for y in range(len_y):
        for x in range(len_x):
            r,g,b,a = img[y][x]
            if a < 128:
                continue
            if empty_matrix[y][x] == 0:
                counter += 1
                found_dict[counter] = Border(y,x)
                # dfs(img,empty_matrix,counter,found_dict, y, x)
                # print (counter, "{},{}:{},{}".format(found_obj[counter].left, found_obj[counter].top, found_obj[counter].right, found_obj[counter].bottom))
                que = queue.Queue()
                que.put((y,x))
                while not que.empty():
                    y1,x1 = que.get()
                    r1,g1,b1,a1 = img[y1][x1]
                    if a1 > 128 and empty_matrix[y1][x1] == 0:
                        empty_matrix[y1][x1] = counter
                        found_dict[counter].check(y1,x1)
                        que.put((y1+1, x1))
                        que.put((y1-1, x1))
                        que.put((y1, x1+1))
                        que.put((y1, x1-1))


def save_sub_image(img, empty_matrix, counter, key, found_obj):
    final_matrix = numpy.ndarray((found_obj.dimension_y, found_obj.dimension_x, 4), dtype=numpy.uint8)
    for y in range(found_obj.dimension_y):
        for x in range(found_obj.dimension_x):
            y1 = found_obj.top + y
            x1 = found_obj.left + x
            if empty_matrix[y1][x1] == key :
                final_matrix[y][x] = img[y1][x1]
            else:
                final_matrix[y][x] = numpy.array([0,0,0,0],dtype=numpy.uint8)
    iio.imwrite("out_"+str(counter)+"_"+sys.argv[1], final_matrix)

if __name__ == "__main__":
    img = iio.v3.imread(sys.argv[1])
    
    counter = 0
    found_obj = {}
    empty_matrix = [ [0] * (len(img[0]) + 1) for y in range(len(img) + 1)]
    find_all_objects(img, empty_matrix, counter, found_obj)
    
    counter = 0
    for key, value in found_obj.items():
        if value.dimension_x > 2 and value.dimension_y > 2:
            save_sub_image(img, empty_matrix, counter, key, value)
            counter +=1

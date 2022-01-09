import os
import cv2
import json
from .fiber import  link
class ThiData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content
        self.mrc=[]
def read_thi(path):
    coordinates = []
    name, _ = os.path.splitext(path)
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('['):
              continue
            content = line.split()
            coordinates.append((float(content[0]), float(content[1])))
    return ThiData(name.split('/')[-1], coordinates)

def read_all_thi(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    thi = []
    for file in os.listdir(path):
        if file.endswith('.thi'):
            #name, _ = os.path.splitext(file)
            #print("Loading %s.thi..." % (name))
            #content = read_thi(path + file)
            thi.append(read_thi(os.path.join(path,file)))
    thi.sort(key=lambda s: s.name)
    return thi

def downsample(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[1]),
            int(coordinates[i][1] * scale[0])
        ))
    return downsampled
'''
def write_thi(inputs, dst):
    print('write_thi')
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for thi_data in inputs:
        print("Writing %s.thi ..." % (thi_data.name))
        with open(dst+thi_data.name+'.thi', "w") as f:
            f.write('[Particle Coordinates: X Y Value]\n')
            for item in thi_data.content:
                f.write("%d\t%d\t%f\n" % (item[0], item[1], item[2]))
            f.write('[End]')
'''
def write_thi(inputs, dst, mode):
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for k in inputs:
        print("Writing %s.thi ..." % (k.name))
        if mode == 'fiber':
            link(k.content, k.mrc,dst+k.name+'.thi',k.name+'.mrc')
        else:
            with open(dst+k.name+'.thi', "w") as f:
              if mode == 'vesicle':
                f.write('[Vesicle Coordinates: X Y Radius Value]\n')
                for item in k.content:
                  f.write("%d\t%d\t%d\t%f\n" % (item[0], item[1], item[2], item[3]))
              elif mode == 'particle':
                f.write('[Particle Coordinates: X Y Value]\n')
                for item in k.content:
                  f.write('%d\t%d\t%f\n' % (item[0], item[1], item[3]))
              f.write('[End]\n')


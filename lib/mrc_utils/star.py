import os     
import cv2
import json
 
class StarData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content

def read_star(path):
    coordinates = []
    name, _ = os.path.splitext(path)

    with open(path) as f:
        x_index, y_index = 0, 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('data_') or line.startswith('loop_') or line.startswith('#'):
                continue
            #_rlnCoordinateX #N means the Nth item stands for x-coordinate
            if line.startswith('_rlnCoordinateX'):
                #relion parameters start from 1, not 0
                #x_index = int(line[len(line)-2]) - 1
                content = line.split()[-1].strip('#')
                x_index = int(content) - 1
                continue
            if line.startswith('_rlnCoordinateY'):
                #y_index = int(line[len(line)-2]) - 1
                content = line.split()[-1].strip('#')
                y_index = int(content) - 1
                continue
            if line.startswith('_rln') or not line.split():
                continue
            content = line.split()
            coordinates.append((int(float(content[x_index])), int(float(content[y_index]))))
            coordinates.sort(key=lambda x: x[0])
    return StarData(name.split('/')[-1], coordinates)

def read_all_star(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    stars = []
    for file in os.listdir(path):
        if file.endswith('.star'):
            #name, _ = os.path.splitext(file)
            #print("Loading %s.star ..." % (name))
            #content = read_star(path + file)
            stars.append(read_star(os.path.join(path, file)))
    stars.sort(key=lambda s: s.name)
    return stars

def downsample_with_size(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[1]),
            int(coordinates[i][1] * scale[0])
        ))
    return downsampled

def write_star(inputs, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not dst.endswith('/'):
        dst += '/'
    #for mrc_data in inputs:
    for star_data in inputs:
        print("Writing %s.star ..." % (star_data.name))
        with open(dst+star_data.name+'.star', "w") as f:
            f.write('\ndata_\n')
            f.write('\nloop_\n')
            f.write('_rlnCoordinateX #1\n')
            f.write('_rlnCoordinateY #2\n')
            f.write('_rlnClassNumber #3\n')
            f.write('_rlnAnglePsi #4\n')
            f.write('_rlnAutopickFigureOfMerit  #5\n')
            for item in star_data.content:
                f.write("%d.0\t%d.0\t-999\t-999.0\t%f\n"%(item[0], item[1], item[3]))
            f.write('\n')

def star2coco(data, root_path, box_size, json_name):
    
    images, categories, annotations = [], [], []
    
    category_dict = {"Particle": 1}
    
    for cat_n in category_dict:
        categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n})
    img_id = 0
    anno_id_count = 0
    for star in data:
        #print(star.name)
        #anno_id_count = 0
        img_name = star.name + '.png'
        img_name = img_name.replace('_autopick','')
        #img_name = img_name.replace('_DW', '')
        img_name = img_name.replace('_manualpick', '')
        img_name = img_name.replace('_empiar', '')
        #print(os.path.join(root_path, img_name))
        img_path = os.path.join(root_path, img_name)
        img_cv2 = cv2.imread(img_path)
        [height, width, _] = img_cv2.shape
        # images info
        images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
        for coord in star.content:
            """
            annotation info:
            id : anno_id_count
            category_id : category_id
            bbox : bbox
            segmentation : [segment]
            area : area
            iscrowd : 0
            image_id : image_id
            """
            category_id = category_dict["Particle"]
            w, h = box_size, box_size
            x1 = float(max(coord[0] - w/2, 1))
            y1 = float(max(coord[1] - h/2, 1))
            x2 = float(min(coord[0] + w/2, width))
            y2 = float(min(coord[1] + h/2, height))
            w = float(w)
            h = float(h)
            bbox = [x1, y1, w, h]
            segment = [x1, y1, x2, y1, x2, y2, x1, y2]
            area = w * h

            anno_info = {'id': anno_id_count, 'category_id': category_id, 'bbox': bbox, 'segmentation': [segment],
                        'area': area, 'iscrowd': 0, 'image_id': img_id}
            annotations.append(anno_info)
            anno_id_count += 1
 
        img_id += 1
 
    all_json = {"images": images, "annotations": annotations, "categories": categories}
    with open(json_name+".json", "w") as outfile:
        json.dump(all_json, outfile)


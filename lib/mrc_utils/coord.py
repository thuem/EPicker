import os
import cv2
import json

class CoordData():
    def __init__(self, name="", content=[]):
        self.name = name
        self.content = content

def read_coord(path):
    coordinates = []
    name, _ = os.path.splitext(path)

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            content = line.split()
            coordinates.append((int(float(content[0])), int(float(content[1]))))
    return CoordData(name.split('/')[-1] ,coordinates)

def read_all_coord(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return None
    if not path.endswith('/'):
        path += '/'
    coords = []
    for file in os.listdir(path):
        if file.endswith('.coord'):
            #name, _ = os.path.splitext(file)
            #print("Loading %s.coord..." % (name))
            #content = read_coord(path + file)
            coords.append(read_coord(os.path.join(path)))
    coords.sort(key=lambda s: s.name)
    return coords

def downsample_with_size(coordinates, scale):
    #scale is a tuple (scale_x, scale_y)
    downsampled = []
    for i in range(len(coordinates)):
        downsampled.append((
            int(coordinates[i][0] * scale[1]),
            int(coordinates[i][1] * scale[0])
        ))
    return downsampled

def write_coord(inputs, path):
  for coord in inputs:
    with open(os.path.join(path, coord.name+'.coord'), 'w') as f:
      for item in coord.content:
        f.write('%d\t%d\t%d\t%d\n' % (item[0], item[1], item[2], item[3]))

def coord2coco(data, json_name):
    root_path = "Falcon_1024/"
    images, categories, annotations = [], [], []
    
    category_dict = {"Falcon": 1}
    
    for cat_n in category_dict:
        categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n})

    img_id = 0
    anno_id_count = 0
    for star in data:
        #anno_id_count = 0
        img_name = star.name + '.png'
        img_name = img_name.replace('_autopick','')
        img_name = img_name.replace('_DW', '')
        img_name = img_name.replace('_manualpick', '')
        img_name = img_name.replace('_empiar', '')
        print(img_name)
        img_cv2 = cv2.imread(root_path + img_name)
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
            category_id = category_dict["Falcon"]
            w, h = 19, 19
            x1 = max(coord[0] - w/2, 1)
            y1 = max(coord[1] - h/2, 1)
            x2 = min(coord[0] + w/2, width)
            y2 = min(coord[1] + h/2, height)

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
'''
if __name__ == '__main__':
    coords = read_all_coord('Falcon_mrc_1024')
    train = coords[0:160]
    valid = coords[160:180]
    test = coords[180:]
    coord2coco(train, 'train')
    coord2coco(valid, 'val')
    coord2coco(test, 'test')
'''
#if __name__ == '__main__':
#   img = cv2.imread('EMPIAR-10017/Falcon_2012_06_13-00_19_28_0.png')
#   coords = read_coord('data/Falcon_2012_06_13-00_19_28_0.coord')
#   box_size = 80
#   boxes = 0
#   for c in coords:
#       x1 = int(c[0] - box_size / 2)
#       y1 = int(c[1] - box_size / 2)
#       x2 = int(c[0] + box_size / 2)
#       y2 = int(c[1] + box_size / 2)
#       img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
#   cv2.imwrite('sample.png', img)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import _init_paths

import json
import os
import cv2
from lib.opts import opts
import copy
def create_exempalr():
    opt = opts().parse()
    N=opt.sampling_size
    print('---------Creating exemplar dataset-------------')
    print('Sampling size:',N)
    anno_dir = os.path.join('./', opt.exp_id, 'annotations/train.json')
    datadir = os.path.join('./', opt.exp_id, 'images/')
    if not os.path.exists(datadir):
        print('No such file or directory:' + dir)
        raise FileNotFoundError
    if not os.path.exists(anno_dir):
        print('No such file or directory:' + anno_dir)
        raise FileNotFoundError
    anno = json.load(open(anno_dir))
    r_dir=opt.load_exemplar
    w_dir = opt.output_exemplar

    num1, num2 = 0, 0
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)
    if not os.path.isfile(os.path.join(r_dir, 'exemplar.json')):
        b = {}
        b['images'] = []
        b['annotations'] = []
        b['categories'] = anno['categories']
    else:
        b = json.load(open(os.path.join(r_dir, 'exemplar.json')))
        num1 = b['images'][-1]['id'] + 1
        num2 = b['annotations'][-1]['id'] + 1
    images_old = []
    for im in b['images']:
        images_old.append(im['file_name'])
    images = copy.deepcopy(anno['images'])
    annotation = anno['annotations']
    j = annotation[N]['image_id']
    print('---------choosing images-------------')
    k = num1
    time = datetime.datetime.now().strftime('%Y%m%d%H')
    for i in range(j + 1):
        file = images[i]['file_name']
        print(file)
        images_name = [i[:-15]+i[-4:] for i in images_old]
        if (file in images_name):
            print('----image has already been written----')
            return None
        img = cv2.imread(datadir + file)
        file=file[:-4]+'_'+time+file[-4:]
        cv2.imwrite(os.path.join(w_dir, file), img)
        anno['images'][i]['id'] = k
        anno['images'][i]['file_name'] = file
        k += 1
        b['images'].append(anno['images'][i])
    total = 0
    for i in range(len(anno['annotations'])):
        if (anno['annotations'][i]['image_id'] <= j):
            total += 1
            anno['annotations'][i]['image_id'] += num1
            anno['annotations'][i]['id'] += num2
            b['annotations'].append(anno['annotations'][i])
    print('choosing', total)
    if(r_dir!=w_dir):
        for im in images_old:
            img = cv2.imread(os.path.join(r_dir,  im))
            cv2.imwrite(os.path.join(w_dir, im), img)
    json.dump(b, open(os.path.join(os.path.join(w_dir, 'exemplar.json')), 'w'))
if __name__ == '__main__':
    create_exempalr()

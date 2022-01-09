import _init_paths
import cv2
import numpy as np
import os
from opts import opts
from mrc_utils.mrc import parse, downsample_with_size, save_image, quantize
def write_thi(point,line_set, filepath):
  with open(filepath, 'w') as f:
    f.write('[Helix Coordinates: X Y Group Value]\n')
    id=1
    for line in line_set:
        for l in line:
            f.write('%d %d %d 1.0\n' % (point[l][0], point[l][1],id))
        id+=1
    f.write('[End]')
def show(line,point,img,s1,s2):
    import random
    def randomcolor():
        color = []
        for i in range(4):
            color += [random.randint(0, 255)]
        return tuple(color)

    for l in line:
        color = randomcolor()
        b,count=l[0],1
        while count <len(l):
            a,b=b,l[count]
            count+=1
            x1, y1 = point[a]
            x2, y2 = point[b]
            cv2.line(img, (int(x1/s1), int(y1/s2)), (int(x2/s1), int(y2/s2)), color, 2)

    return img
def complex_gen(x,y):
    x1, y1 = x
    x2, y2 = y
    xx = x2 - x1
    yy = y2 - y1
    return complex(xx, yy)
def compute_dis_matrix(point):
    x = point[:, 0].reshape(-1, 1)
    X = np.tile(x, x.shape[0])
    X = X - X.T
    y = point[:, 1].reshape(-1, 1)
    Y = np.tile(y, y.shape[0])
    Y = Y - Y.T
    dis_matrix = np.sqrt(X ** 2 + Y ** 2)
    return dis_matrix
def fiber_connect(point,line_sign,dis_matrix,cpoint,num,pre_angle,min_dis=120,ang_thresh=0.3):
    cur_angle=None
    if pre_angle:
        cur_angle=-pre_angle
    i = 0
    record = []
    while i < len(dis_matrix) and line_sign.count(-1)!=0:
        if not cur_angle:
            dis_matrix[cpoint][cpoint] = float('inf')
            min_point = dis_matrix[cpoint].argmin()
            if dis_matrix[cpoint].min() < min_dis and line_sign[min_point] == -1:
                line_sign[cpoint] = line_sign[min_point] = num
                record.extend([cpoint, min_point])
                num += 1
                cur_angle=complex_gen(point[cpoint], point[min_point])
                cpoint = min_point
                i = 0
                continue
            else:
                dis_matrix[cpoint][min_point] = float('inf')
        else:
            dis_matrix[cpoint][cpoint] = float('inf')
            if dis_matrix[cpoint].min() > min_dis:
                break
            min_point = dis_matrix[cpoint].argmin()
            if dis_matrix[cpoint][min_point]< min_dis and line_sign[min_point] == -1:
                tmp1 = cur_angle
                tmp2 = complex_gen(point[cpoint], point[min_point])
                ang = np.angle(tmp1 / (tmp2+1e-5))
                if abs(ang) < ang_thresh:
                    line_sign[min_point] = line_sign[cpoint]
                    cur_angle=tmp2# +cur_angle
                    cpoint = min_point
                    i = 0
                    record.append(cpoint)
                    continue
                else:
                    dis_matrix[cpoint][min_point] = float('inf')

            else:
                dis_matrix[cpoint][min_point] = float('inf')
        i += 1
    return record,cur_angle
def fiber_main(point,min_dis=100,ang_thresh=0.3,min_particle=2):
    dis_matrix = compute_dis_matrix(point)
    line_record=[]
    num = 0
    line_sign = [-1 for _ in range(len(point))]
    cpoint = 0
    line_sign[0] = 1
    while line_sign.count(-1)>1:
        record1,pre_angle=fiber_connect(point,line_sign, dis_matrix, cpoint, num,None,min_dis=min_dis,ang_thresh=ang_thresh)
        record2=fiber_connect(point,line_sign, dis_matrix, cpoint, num,pre_angle,min_dis=min_dis,ang_thresh=ang_thresh)[0][::-1]
        record=record2+record1
        if(line_sign[cpoint]==-1):
            line_sign[cpoint]=-2
        for (i,n) in enumerate(line_sign):
            if(n==-1):
                cpoint=i
                break
        if record and len(record)>min_particle:
            line_record.append(record)
    return line_record
def post_process(lines,point,ang_thresh=0.15):
    for j in range(len(lines)):
        line=lines[j]
        start_point = None
        line_new=[]
        for i in range(len(line)):
            if start_point==None:
                pre_point=start_point=line[0]
                pre_v=complex_gen(point[pre_point], point[line[1]])
                line_new.extend([line[0],line[1]])
                continue
            else:
                pre_point=line[i-1]
                cur_v=complex_gen(point[pre_point], point[line[i]])
            ang = np.angle(pre_v / (cur_v+1e-5))
            if abs(ang) < ang_thresh:
                line_new[-1]=line[i]
                pre_v += cur_v
            else:
                line_new.append(line[i])
                if i!=(len(line)-1):
                    pre_v=complex_gen(point[line[i]], point[line[i+1]])

        lines[j]=line_new
    return lines
def link(point,size_mrc,path,img_path):
    opt = opts().parse()
    ang_thresh=opt.ang_thresh

    point = np.array(point)[:,:2]
    line_record=fiber_main(point,min_dis=size_mrc[0]/10,ang_thresh=ang_thresh,min_particle=4)
    line_record=post_process(line_record,point,ang_thresh=0.15)
    write_thi(point,line_record,path)
    if opt.visual:
        with open(os.path.join(opt.data, img_path), "rb") as f:
            content = f.read()
        data, header, _ = parse(content=content)
        if header[2] > 1:
            temp = np.zeros(data[0].shape)
            for i in range(header[2]):
                temp += data[i, ...]
            data = temp / header[2]
        data = downsample_with_size(data, int(data.shape[0] / data.shape[1] * 1024), 1024)
        data = quantize(data)
        data = np.expand_dims(data, axis=-1)
        data = cv2.equalizeHist(data)
        data = cv2.merge([data, data, data])
        img = show(line_record, point, data,size_mrc[1]/data.shape[0],size_mrc[0]/data.shape[1])
        # cv2.imshow('test',img)
        # cv2.waitKey()
        cv2.imwrite(os.path.splitext(path)[0]+'.png', img)

if __name__ == '__main__':
  #img_path=''
  #img_anno=''
  #img_anno_new='/'.join(img_anno.split('/')[:-1])+'/'+img_anno.split('/')[-1].split('.')[:-1][0]+'_trace.thi'
  opt = opts().parse()
  image_names = [f for f in os.listdir(opt.data) if f.endswith('mrc')]
  img_anno=[f for f in os.listdir(opt.label) if f.endswith('thi')]
  for img_path in image_names:
      if(os.path.splitext(img_path)[0]+'.thi') not in img_anno:
          continue
      anno_path=os.path.splitext(img_path)[0]+'.thi'
      with open(os.path.join(opt.data,img_path), "rb") as f:
          content = f.read()
      data, header, _ = parse(content=content)
      if header[2] > 1:
          temp = np.zeros(data[0].shape)
          for i in range(header[2]):
              temp += data[i, ...]
          data = temp / header[2]
      print('Loading', img_path, '...')
      ori_w,ori_h=data.shape
      data = downsample_with_size(data, int(data.shape[0] / data.shape[1] * 1024), 1024)
      data = quantize(data)
      data = np.expand_dims(data, axis=-1)
      data = cv2.equalizeHist(data)
      data = cv2.merge([data, data, data])
      link(os.path.join(opt.label,anno_path), opt, data,ang_thresh=opt.ang_thresh,w=ori_w,h=ori_h,path=os.path.join(opt.output,anno_path))

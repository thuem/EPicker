import os

def read_thi(path):
  group = {}
  with open(path) as f:
    while (True):
      line = f.readline()
      if not line:
        break
      if line.startswith('['):
        continue
      content = line.split()
      x, y, group_id = int(float(content[0])), int(float(content[1])), content[2]
      if not group_id in group:
        group[group_id] = [(x, y)]
      else:
        group[group_id].append((x,y))
  return group

def place_boxes(group, dis=5):
  boxes = []
  for group_id in group:
    nodes = group[group_id]
    for i in range(len(nodes)-1):
      boxes.append(nodes[i])
      x1, y1 = nodes[i][0], nodes[i][1]
      x2, y2 = nodes[i+1][0], nodes[i+1][1]
      if abs(x2 - x1) > abs(y2 - y1):
        if x1 == x2:
          x1 += 1
        if x1 >= x2:
          for x in range(x2, x1,dis):
            y = int((x - x1)*(y2 - y1)/(x2 - x1)) + y1
            boxes.append((x,y))
        else:
          for x in range(x1, x2,dis):
            y = int((x - x1)*(y2 - y1)/(x2 - x1)) + y1
            boxes.append((x,y))
      else:
        if y1 == y2:
          y1 += 1
        if y1 >= y2:
          for y in range(y2, y1,dis):
            x = int((y - y1)*(x2 - x1)/(y2 - y1)) + x1
            boxes.append((x,y))
        else:
          for y in range(y1, y2,dis):
            x = int((y - y1)*(x2 - x1)/(y2 - y1)) + x1
            boxes.append((x,y))
  return boxes

def write_thi(boxes, filepath):
  with open(filepath, 'w') as f:
    f.write('[Helix X Y Value]\n')
    for box in boxes:
      f.write('%d %d 1.0\n' % (box[0], box[1]))
    f.write('[End]')

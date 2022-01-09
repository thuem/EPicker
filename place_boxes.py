
import os
import cv2 as cv
import argparse

from gen_box import read_thi, place_boxes, write_thi

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')

args = parser.parse_args()
path = args.input
output_path = args.output

for d in os.listdir(path):
  if d.endswith('.thi'):
    group = read_thi(os.path.join(path,d))
    boxes = place_boxes(group)
    write_thi(boxes, os.path.join(output_path,d))

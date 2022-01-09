from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger
from mrc_utils.thi import write_thi
from mrc_utils.box import write_box
from mrc_utils.coord import write_coord
from mrc_utils.star import write_star
from mrc_utils.thi import ThiData
from .base_detector import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, opt, return_time=False):
        with torch.no_grad():
            output, fea = self.model(images)
            output = output[-1]
            hm = output['hm'].sigmoid_()
            if opt.mode == 'particle' or opt.mode == 'fiber':
                if (opt.gpus[0] == 0):
                    wh = torch.Tensor(output['wh'].shape).fill_(float(opt.particle_size)).to('cuda')
                else:
                    wh = torch.Tensor(output['wh'].shape).fill_(float(opt.particle_size))
            elif opt.mode == 'vesicle':
                wh = output['wh']
            # elif opt.mode == 'fiber':
            #   return NotImplementedError
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.2, method=0)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results, image_name, header=None):
        debugger.add_img(image, img_id=image_name_name.split('/')[-1].replace('mrc',''))
        name = image_name.split('/')[-1].replace('.png', '')
        thi = ThiData(name=name, content=[])
        thi.mrc = [header[0], header[1]]
        boxes = 0
        if header == None:
            imsize = (1024, 1024)
            dis = (self.opt.edge, self.opt.edge)
        else:
            imsize = (1024, header[1] / header[0] * 1024)
            dis = (self.opt.edge, header[1] / header[0] * self.opt.edge)
        min_dis = self.opt.min_distance
        for j in range(1, self.num_classes + 1):
            result_new = []
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    if not self.bbox_valid(bbox, imsize, dis):
                        continue
                    else:
                        result_new.append(bbox)
            results[j] = self.distance_valid(result_new, min_dis)

            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=image_name.split('/')[-1].replace('mrc', ''))
                    if self.opt.data_type == 'mrc':
                        size = 0
                        if self.opt.mode == 'particle' or self.opt.mode == 'fiber':
                            size = self.opt.particle_size * 4
                        elif self.opt.mode == 'vesicle':
                            size = (bbox[3] + bbox[2] - bbox[1] - bbox[0]) / 4 * header[0] / 1024
                        # elif self.opt.mode == 'fiber':
                        #   return NotImplementedError

                        if self.opt.output_type == 'thi' or self.opt.output_type == 'star' or self.opt.output_type == 'coord':
                            thi.content.append(
                                ((bbox[0] + bbox[2]) / 2 * header[0] / 1024,
                                 (bbox[1] + bbox[3]) / 2 * header[0] / 1024,
                                 size,
                                 bbox[4])
                            )
                        else:
                            thi.content.append(
                                (bbox[0] * header[0] / 1024,
                                 bbox[1] * header[0] / 1024,
                                 bbox[2] * header[0] / 1024,
                                 bbox[3] * header[0] / 1024)
                            )
                    elif self.opt.data_type == 'png':
                        if self.opt.output_type == 'thi' or self.opt.output_type == 'star' or self.opt.output_type == 'coord':
                            thi.content.append(
                                ((bbox[0] + bbox[2]) / 2,
                                 (bbox[1] + bbox[3]) / 2,
                                 (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2,
                                 bbox[4])
                            )
                        else:
                            thi.content.append(
                                (bbox[0], bbox[1], bbox[2], bbox[3])
                            )
                    boxes += 1
        print(boxes, ' objects picked')
        img_path = self.output_dir
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if self.opt.visual == True and self.opt.mode != 'fiber':
            debugger.save_all_imgs(path=img_path, genID=False)
        if self.opt.data_type == 'mrc':
            output_path = self.output_dir
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if self.opt.output_type == 'thi':
                write_thi([thi], output_path, self.opt.mode)
            elif self.opt.output_type == 'star':
                write_star([thi], output_path)
            elif self.opt.output_type == 'box':
                write_box([thi], output_path)
            elif self.opt.output_type == 'coord':
                write_coord([thi], output_path)
            else:
                print('Invalid output type: should be thi | star | box | coord')

    def bbox_valid(self, bbox, imsize, dis):
        if bbox[0] < dis[0] or bbox[2] > imsize[0] - dis[0]:
            return False
        elif bbox[1] < dis[1] or bbox[3] > imsize[1] - dis[1]:
            return False
        else:
            return True

    def distance_valid(self, results, min_dis=40):
        results = np.array(results)
        c1 = (results[:, 0] + results[:, 2]) / 2
        c2 = (results[:, 1] + results[:, 3]) / 2
        keep = []
        scores = results[:, 4]
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]
            keep.append(i)
            distance = (c1[i] - c1[index[1:]]) ** 2 + (c2[i] - c2[index[1:]]) ** 2
            idx = np.where(distance > min_dis ** 2)[0]
            index = index[idx + 1]
        return results[keep]

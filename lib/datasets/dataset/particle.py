from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
#from sklearn import metrics

class Particle(data.Dataset):
    num_classes = 1
    default_resolution = [1024, 1024]
    mean = np.array([0.508145, 0.508145, 0.508145],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.288813, 0.288813, 0.288813],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(Particle, self).__init__()
        opt.data_dir = './'
        if opt.data_type == 'mrc':
            self.data_dir = os.path.join(opt.data_dir, opt.exp_id)
            self.img_dir = os.path.join(self.data_dir, 'images')
        elif opt.data_type == 'png':
            self.data_dir = opt.data
            self.img_dir = os.path.join(self.data_dir, 'images')
        if split == 'val':
            sign = 1
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'val.json')
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'train.json')
            if split == 'test':
                sign = 1
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'test.json'
                )
            elif split == 'train':
                sign = 1
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'train.json')
            else:
                sign = 0
                split = 'train'
                self.img_dir = os.path.join(opt.load_exemplar)
                self.annot_path = os.path.join(opt.load_exemplar,
                                               'exemplar.json')
        self.max_objs = 1500
        self.class_name = [
            '__background__', 'particle']
        self._valid_ids = [
            1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


        self.split = split
        self.opt = opt
        self.sign = sign

        print('==> initializing {} {} data.'.format(opt.exp_id, split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def bbox_valid(self, bbox, imsize=1024, dis=5):
        if (bbox[0] + (bbox[2] / 2)) < dis or (bbox[0] + (bbox[2] / 2)) > imsize - dis:
            return False
        elif (bbox[1] + (bbox[3] / 2)) < dis or (bbox[1] + (bbox[3] / 2)) > imsize - dis:
            return False

        else:
            return True

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        dis = 2 * self.opt.particle_size
        self.save_results(results, save_dir)
        a = json.load(open('{}/results.json'.format(save_dir)))
        for i in range(len(a) - 1, -1, -1):
            if not self.bbox_valid(a[i]['bbox'],dis=dis):
                del (a[i])
        json.dump(a, open('{}/processed_results.json'.format(save_dir), 'w'))
        coco_dets = self.coco.loadRes('{}/processed_results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.params.maxDets = [700, 900, 1000]
        coco_eval.params.iouThrs = np.array([0.5])#np.linspace(.5, 0.8, int(np.round((0.8 - .5) / .05)) + 1, endpoint=True)
        coco_eval.params.areaRng = [[0, 1e5 ** 2], [0, 1e5 ** 2], [0, 1e5 ** 2], [0, 1e5 ** 2]]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        pr1 = coco_eval.eval['precision'][0, :, 0, :, 2]

        x = np.arange(0.0, 1.01, 0.01)
        plt.switch_backend('agg')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.plot(x, pr1, 'r-',label = 'Precision-Recall curve')
        handles, labels = plt.gca().get_legend_handles_labels()
        from collections import OrderedDict
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc='lower left',)
        plt.savefig('./PR.jpg')

        # tps, fps = coco_eval.accumulate()
        # tps = [tps[i] for i in range(0, len(tps), 50)]
        # fps = [fps[i] for i in range(0, len(fps), 50)]
        # auc=metrics.auc(fps,tps)
        # print(' AUC:',auc)
        # plt.switch_backend('agg')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.xlim(0, 1.0)
        # plt.ylim(0, 1.0)
        # plt.grid(True)
        # plt.plot(fps, tps, 'b-',label = 'ROC curve (AUC = '+ str(np.round(auc,3)) +')')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # from collections import OrderedDict
        # by_label = OrderedDict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(),loc='lower right',)
        # plt.savefig('./AUC.jpg')


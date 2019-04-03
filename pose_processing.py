from typing import List
from utils import prep_image, crop_from_dets, vis_frame_fast, vis_frame
from yolo.darknet import Darknet
from pPose_nms import pose_nms
import os
import cv2
from SPPE.src.utils.img import im_to_torch
import torch
from yolo.util import dynamic_write_results

from dataloader import Mscoco
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction


class PoseProcessing:
    def __init__(self, use_gpu=True, fast_inference=True):
        self.gpu = use_gpu

        # load detection model
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = 608
        self.det_inp_dim = int(self.det_model.net_info['height'])

        # Load pose model
        pose_dataset = Mscoco()
        if fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)

        if use_gpu:
            self.pose_model.cuda()
            self.det_model.cuda()
        self.pose_model.eval()
        self.det_model.eval()

    def det_process(self, orig_im):
        img, orig_img, im_dim_list = prep_image(orig_im, inp_dim=608)
        with torch.no_grad():
            orig_img, boxes, scores, inps, pt1, pt2 = self.detection(img, orig_img, im_dim_list)
            if boxes is None or boxes.nelement() == 0:
                return None, orig_img, boxes, scores, None, None

            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

            return inps, orig_img, boxes, scores, pt1, pt2

    def detection(self, img, orig_img, im_dim_list):
        with torch.no_grad():
            # For Human Detection
            img = torch.cat([img])
            im_dim_list = torch.FloatTensor([im_dim_list]).repeat(1, 2)
            if self.gpu:
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
            else:
                prediction = self.det_model(img, CUDA=False)
            # NMS process
            dets = dynamic_write_results(prediction, 0.05, 80, nms=True, nms_conf=0.6)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return orig_img, None, None, None, None, None

            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return orig_img, None, None, None, None, None

        inps = torch.zeros(boxes_k.size(0), 3, 320, 256)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)

        return orig_img, boxes_k, scores[dets[:, 0] == 0], inps, pt1, pt2

    def pose_detection(self, imgs: List[np.array], img_names: List[str], output_path: str, vis=True):
        for i in range(len(imgs)):
            img = imgs[i]

            # do person detection
            with torch.no_grad():
                inps, orig_img, boxes, scores, pt1, pt2 = self.det_process(img)
                if boxes is None or boxes.nelement() == 0:
                    print('No person Detected')
                    continue
                # Pose Estimation
                inps_j = inps.cuda()
                hm_j = self.pose_model(inps_j)

            hm_data = hm_j.cpu()
            preds_hm, preds_img, preds_scores = getPrediction(hm_data, pt1, pt2, 320, 256, 80, 64)
            result = pose_nms(boxes, scores, preds_img, preds_scores)
            if vis:
                img_name = img_names[i]
                out_path = os.path.join(output_path, img_name)
                out_img = vis_frame(orig_img, result)
                cv2.imwrite(out_path, out_img)
            print(result)

    def process(self, input_path, output_path, vis=True):
        imgs = os.listdir(input_path)
        imgs_path = []
        imgs_array = []
        names = []

        for img in imgs:
            names.append(img)
            path = os.path.join(input_path, img)
            imgs_path.append(path)
            imgs_array.append(cv2.imread(path))

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.pose_detection(imgs_array, names, output_path, vis=vis)


if __name__ == "__main__":
    pp = PoseProcessing()
    in_path = '/media/haoxue/WD/Fall-Detection-Research/datasets/fall/fall-30-cam0-rgb'
    o_path = 'test'
    pp.process(input_path=in_path, output_path=o_path)



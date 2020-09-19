# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import torch
import torch.utils.data as data
import torchvision

import os
import json
import random

import numpy as np
from numpy.random import randint
from PIL import Image

class InstanceRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_sec(self):
        return self._data[1]

    @property
    def end_sec(self):
        return self._data[2]

    @property
    def label(self):
        return int(self._data[3])

    def __str__(self):
        return '%s, %.3f, %.3f, %d'%(self.path, self.start_sec, self.end_sec, self.label)


class TALDataSet(data.Dataset):
    def __init__(self, root_path, meta_dict, dataset,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 dense_sample=False, twice_sample=False):
        '''meta_dict: set of metadata which is needed to parse dataset
        e.g) class_idx, annotation file'''
        print ('-'*20, 'Start init.', '-'*20)
        self.root_path = root_path
        self.meta_dict = meta_dict
        self.dataset = dataset
        assert self.dataset in ['Thumos', 'ANet', 'BBDB'], 'Wrong dataset: %s'%(self.dataset)
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._prepare_data()  # Parse all existing action instances and Save it as a list of [vid_path, start_sec, end_sec, class] in self.instances
        print ('-'*20, 'Finish init.', '-'*20)

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_thumos(self):
        '''return
        1. idx_to_class
        2. list of [path, start_sec, end_sec, label]'''

        # Parse action list
        idx_to_class = [x.strip() for x in open(self.meta_dict['classidx'])]  # BG at index 0

        # Parse instances
        annots = []
        action_files = [os.path.join(self.meta_dict['annt_path'], file) for file in os.listdir(self.meta_dict['annt_path']) if file.endswith('.txt')]
        for action_file in action_files: 
            action = action_file.split('/')[-1].split('_')[0]
            if action == 'Ambiguous': continue
            with open(action_file,'r') as f:
                for line in f:
                    vid = line.split(' ')[0]
                    st = float(line.split('  ')[1].split(' ')[0])
                    et = float(line.split('  ')[1].split(' ')[1][:-1])
                    # print(vid, st, et )
                    path = os.path.join(self.root_path, vid + '.mp4')
                    assert os.path.exists(path), '%s does not exist'%(path)
                    label = idx_to_class.index(action)
                    annot = [path, st, et, label]
                    annots.append(annot)

        print('Instance number:%d' % (len(annots)))
        return idx_to_class, annots

    def _parse_bbdb(self):
        idx_to_class = [x.strip() for x in open(self.meta_dict['classidx'])]

        annots = []
        # list of [path, start_sec, end_sec, label]

        subset = self.meta_dict['subset']
        assert subset in ('train', 'val', 'otal_test'), 'Unknown BBDB Subset!! (train | val | test | otal_test)'
        # TODO
        

    def _parse_anet(self):
        '''
        meta_dict elements:
        1. subset (training | validation | otal_test)
        2. annt_file (json file path)
        3. classidx (class list text file path)
        '''

        idx_to_class = [x.strip() for x in open(self.meta_dict['classidx'])]

        annots = []
        # list of [path, start_sec, end_sec, label]

        subset = self.meta_dict['subset']
        assert subset in ('training', 'validation', 'otal_test'), 'Unknown Anet Subset!! (training | validation | otal_test)'
        # training / validation -> use official JSON annotation file for training & validation
        # otal_test -> use JSON file generated by OTAL model

        if subset in ('training', 'validation'):
            with open(self.meta_dict['annt_file'], 'r') as json_file:
                json_data =  json.load(json_file)
                database = json_data["database"]

                for video_name in database:
                    path = os.path.join(self.root_path, 'v_' + video_name + '.mp4')
                    if os.path.exists(path) == False or database[video_name]["subset"] != subset:
                        continue
                    
                    video_info = database[video_name]
                    for segment in video_info["annotations"]:
                        st = segment["segment"][0]
                        et = segment["segment"][1]
                        if st == et and subset == "training":
                            continue

                        label = idx_to_class.index(segment["label"])

                        annot = [path, st, et, label]
                        annots.append(annot)
                    
        elif subset is 'otal_test':   # otal_test : test with OTAL RL output
            with open(self.meta_dict['annt_file'], 'r') as json_file:
                json_data = json.load(json_file)
                database = json_data["results"]

                for video_name in database:
                    path = os.path.join(self.root_path, 'v_' + video_name + '.mp4')
                    if os.path.exists(path) == False:
                        continue
                    
                    video_info = database[video_name]
                    for segment in video_info:
                        st = segment["segment"][0]
                        et = segment["segment"][1]
                        label = -1

                        annot = [path, st, et, label]
                        annots.append(annot)

        return idx_to_class, annots

    def _prepare_data(self):
        if self.dataset == 'Thumos':
            idx_to_class, annots = self._parse_thumos()
        elif self.dataset == 'BBDB':
            idx_to_class, annots = self._parse_bbdb()
        elif self.dataset == 'ANet':
            idx_to_class, annots = self._parse_anet()
        '''
        annots = [x.strip().split(' ') for x in open(self.meta_dict)]
        # check the frame number is large >3:
        if not self.test_mode:
            annots = [item for item in annots if int(item[1]) >= 3]
        '''
        ## TODO
        # What if #len == 0?
        
        self.idx_to_class = idx_to_class
        self.class_to_idx = {}
        for idx, classname in enumerate(self.idx_to_class):
            self.class_to_idx[classname] = idx

        self.instances = [InstanceRecord(item) for item in annots]
        
        print('Class', self.idx_to_class)
        print('Instance number:%d' % (len(self.instances)))

    def _sample_indices(self, num_fr):
        """
        :param num_fr: num_fr in the instance
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_fr - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_fr for idx in range(self.num_segments)]
            return np.array(offsets)  # + 1
        else:  # normal sample
            average_duration = (num_fr - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif num_fr > self.num_segments:
                offsets = np.sort(randint(num_fr - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets  # + 1

    def _get_val_indices(self, num_fr):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_fr - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_fr for idx in range(self.num_segments)]
            return np.array(offsets)  # + 1
        else:
            if num_fr > self.num_segments + self.new_length - 1:
                tick = (num_fr - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets  # + 1

    def _get_test_indices(self, num_fr):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_fr - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % num_fr for idx in range(self.num_segments)]
            return np.array(offsets)  # + 1
        elif self.twice_sample:
            tick = (num_fr - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets  # + 1
        else:
            tick = (num_fr - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets  # + 1

    def _endpoint_boundary(self, org_s, org_e, start_point):
        if start_point < org_s:
            end_s = 0.5 * (org_e - start_point) + org_s
            end_e = 2 * (org_e - org_s) + start_point
        elif start_point == org_s:
            end_s = 0.5 * (org_s + org_e)
            end_e = org_e + (org_e - org_s)
        elif start_point > org_s:
            end_s = 0.5 * (org_e - org_s) + start_point
            end_e = 2 * (org_e - start_point) + org_s

        return end_s, end_e

    def _temporal_jittering(self, org_s, org_e):
        '''TODO: when BBDB.. more constrains on endpoint
        '''
        len_seg = org_e - org_s

        # randomly pick start point
        start_s = max(0, org_s - len_seg)
        start_e = org_s + (len_seg / 2.0)
        random_prob = random.random()
        if random_prob < 0.1:  # 1:9
            start_point = random.uniform(start_s, org_s)
        else:
            start_point = random.uniform(org_s, start_e)

        # calculate end point boundary depending on the selected start_point
        end_s, end_e = self._endpoint_boundary(org_s, org_e, start_point)

        # randomly pick end point
        end_point = random.uniform(end_s, end_e)

        return start_point, end_point

    def _cal_IoU(self, s1, e1, s2, e2):
        start = max(s1, s2)
        end = min(e1, e2)
        intersection = max(0, end - start)
        union = (e1 - s1) + (e2 - s2) - intersection

        if union != 0:
            tIoU = intersection * 1.0 / union
        else:
            tIoU = 0

        return tIoU

    def __getitem__(self, index):
        original_record = self.instances[index]  # Original instance

        # Check this is a legit video folder
        assert os.path.exists(original_record.path), '%s does not exists'%(original_record.path)

        # Temporal augmentation (train/val only?)
        # Generate a jittered instance while maintaining IoU >= 0.5
        if not self.test_mode:  # train or validation
            if self.random_shift:  # train
                org_s, org_e = original_record.start_sec, original_record.end_sec
                new_s, new_e = self._temporal_jittering(org_s, org_e)
                assert self._cal_IoU(org_s, org_e, new_s, new_e) >= 0.5, 'IoU smaller than 0.5, (%.2f, %.2f, %.2f, %.2f)'%(org_s, org_e, new_s, new_e)
                record = InstanceRecord([original_record.path, new_s, new_e, original_record.label])
            else:  # validation
                record = original_record
        else:  # test
            record = original_record

        # Retrieve frames of instances
        frs_pt = self.get_fr(record)
        num_fr = frs_pt.size(0)
        assert num_fr>0, 'num_fr is 0, %s'%(record)
        # torchvision.io.write_video('tmp.mp4',frs_pt,25)  # Sanity check

        if not self.test_mode:  # train or validation
            if self.random_shift:  # train
                segment_indices = self._sample_indices(num_fr)
            else:  # validation
                segment_indices = self._get_val_indices(num_fr)
        else:  # test
            segment_indices = self._get_test_indices(num_fr)

        # Retrieve only selected frames
        segment_indices = segment_indices.astype(np.int64)
        segment_indices = torch.as_tensor(segment_indices)
        frs_pt = torch.index_select(frs_pt, dim=0, index=segment_indices)

        # Spatial augmentation
        # Before spatial augmentation, we need to convert pytorch tensor to list of PIL to use the existing augmentation functions
        frs_np = frs_pt.numpy()  # (t, h, w, c)  # Assume that channel is order in 'RGB'
        frs_PIL = [Image.fromarray(fr) for fr in frs_np]  # size (w, h)
        process_data = self.transform(frs_PIL)  # (t * 3, h, w)

        return process_data, record.label

        # return self.get(record, segment_indices)

    def get_fr(self, record):
        vid_pt, _, _ = torchvision.io.read_video(record.path, record.start_sec, record.end_sec, pts_unit='sec')  # vid, aud, info
        # vid_pt: Pytorch Tensor (t, h, w, c)

        return vid_pt

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.instances)

if __name__ == '__main__':
    from transforms import *

    ANET_CLASSIDX = '/workspace/ActivityNet_200_1.3/anet_classidx.txt'
    ANET_ANNOTATION_FILE = '/workspace/ActivityNet_200_1.3/rl_annotation.json'
    ANET_VID_PATH = '/workspace/ActivityNet_200_1.3/videos/'

    anet_dataset = TALDataSet(ANET_VID_PATH, 
                   {'classidx':ANET_CLASSIDX, 'annt_file':ANET_ANNOTATION_FILE, 'subset':"otal_test"},
                   'ANet', 
                   num_segments=8,
                   new_length=1,
                   modality='RGB',
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(256)),
                       GroupCenterCrop(224),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                   ]), dense_sample=False)

    for item in data.DataLoader(anet_dataset, batch_size=1, shuffle=True):
        data, target = item
        print (data.size(), target)
        # print("video name: ", item['video_name'])
        # print("feature shape : ", item['data'].shape)
        # print("label seq shape : ", item['label'].shape)
        input()
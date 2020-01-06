from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.ctdet_pku import CTDetPkuDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.pku_dataset import PkuDataset


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'pku': PkuDataset,
}

_sample_factory = {
  'exdet': EXDetDataset,
  # 'ctdet': CTDetDataset,
  'ctdet': CTDetPkuDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  # 'pku': PkuSample
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  

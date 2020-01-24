from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ctdet_pku import CtdetPkuTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdet_pku': CtdetPkuTrainer,
  'multi_pose': MultiPoseTrainer, 
  'multi_pose_pku': MultiPoseTrainer
}

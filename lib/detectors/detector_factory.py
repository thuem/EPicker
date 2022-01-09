from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector

detector_factory = {
  'ctdet': CtdetDetector,
}

import numpy as np
import sklearn.metrics
from utils.utils import threshold_logits


def f1_score(actual, predicted, average='macro'):
  actual = np.array(actual)
  predicted = np.array(predicted)
  return sklearn.metrics.f1_score(actual, predicted, average=average)


def kappa_loss(y1, y2, coef=None):

  ll = sklearn.metrics.cohen_kappa_score(y1, y2, weights='quadratic')
  return ll

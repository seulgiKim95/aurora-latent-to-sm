import datetime
import numpy as np
from abc import *


def get_now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def calc_vpd(t, q, p=100, absolute_temperature=False):
    """calculate vapor pressure deficit

    Args:
        t (`numpy.ndarray`): temperature (2-m), Celsius
        q (`numpy.ndarray`): specific humidity (2-m ~ 100 kPa = 1000 hPa)
        p (`float`, optional): pressure (100 kPa at 2-m)
        absolute_temperature (`bool`, optional): use absolute temperature

    Returns:
        `numpy.ndarray`: VPD
    """
    if absolute_temperature:
        temp = t - 273.15
    e_sat = 0.6108 * np.exp(17.28 * temp / (temp + 237.3))
    e_a = q * p / (0.622 + q)
    vpd = e_sat - e_a
    return vpd


class AbstractTransformer():
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transformer():
        pass


class LogTransformer(AbstractTransformer):
    def __init__(self, a, eps):
        self.a = a
        self.eps = eps

    def transform(self, X):
#       return np.log(self.a * (X + self.eps))
        return np.log(1. + X / self.eps)

    def inverse_transform(self, X):
#       return np.exp(X) / self.a - self.eps
        return (np.exp(X) - 1.) * self.eps


class HyperbolicArcSineTransformer(AbstractTransformer):
    def __init__(self, eps):
        self.eps = eps

    def transform(self, X):
        return np.arcsinh(X / self.eps)

    def inverse_transform(self, X):
        return np.sinh(X) * self.eps


def parse_datetime(dt_str):
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H',
        '%Y-%m-%d'
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
            
    raise ValueError(f"undefined format '{dt_str}'")
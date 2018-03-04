import warnings

import six
import abc

@six.add_metaclass(abc.ABCMeta)
class Function(object):

    def __init__(self):
        self.test()

    @abc.abstractmethod
    def function(self, x):
        pass

    def getDescription(self):
        return self.__class__.__name__ + " function"

    @abc.abstractmethod
    def getBounds(self):
        pass

    def test(self):
        warnings.warn("No tests specified for " + self.getDescription())
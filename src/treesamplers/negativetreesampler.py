import numpy as np

from .treesampler import TreeSampler

class NegativeTreeSampler(object):

    def __init__(self):
        self.sampler_no = TreeSampler(["X", "X", "BU"])
        self.sampler_sonly = TreeSampler(["BU", "X", "BU"])
        self.sampler_ponly = TreeSampler(["X", "BU", "BU"])
        self.sampler_sp = TreeSampler(["BU", "BU", "BU"])

    def sample(self, inputs, sbnds, pbnds):
        """
        :type inputs: list of int
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :rtype: list of str
        """
        selected = np.random.choice(["no", "sonly", "ponly", "sp"],
                                    p=[0.4, 0.6, 0.0, 0.0])
        if selected == "no":
            sexp = self.sampler_no.sample(
                                inputs=inputs,
                                edus=None,
                                edus_head=None,
                                sbnds=sbnds,
                                pbnds=pbnds)
        elif selected == "sonly":
            sexp = self.sampler_sonly.sample(
                                inputs=inputs,
                                edus=None,
                                edus_head=None,
                                sbnds=sbnds,
                                pbnds=pbnds)
        elif selected == "ponly":
            sexp = self.sampler_ponly.sample(
                                inputs=inputs,
                                edus=None,
                                edus_head=None,
                                sbnds=sbnds,
                                pbnds=pbnds)
        else:
            sexp = self.sampler_sp.sample(
                                inputs=inputs,
                                edus=None,
                                edus_head=None,
                                sbnds=sbnds,
                                pbnds=pbnds)
        return sexp


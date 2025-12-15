#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchwrench.extras.hdf import HDFDataset


class HDFAACDataset(HDFDataset):
    def at(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

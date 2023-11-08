#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoNeTTE model for Audio Captioning.
"""

__name__ = "conette"
__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.1.0"


from conette.nn.huggingface import (  # noqa: F401
    CoNeTTEConfig,
    CoNeTTEModel,
    get_sample_path,
)

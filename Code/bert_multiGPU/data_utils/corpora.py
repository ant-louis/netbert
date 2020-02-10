# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset
import os


class cisco_train(json_dataset):
    """
    dataset for cisco_train with arguments configured for convenience

    command line usage: `--train-data cisco_train`
    """
    PATH = '/raid/antoloui/Master-thesis/Data/Cleaned/New_cleaning/train.json'
    assert_str = "make sure to set PATH for cisco_train data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert os.path.exists(cisco_train.PATH), \
                        cisco_train.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(cisco_train, self).__init__(cisco_train.PATH, **kwargs)

        
class cisco_dev(json_dataset):
    """
    dataset for cisco_dev with arguments configured for convenience

    command line usage: `--train-data cisco_dev`
    """
    PATH = '/raid/antoloui/Master-thesis/Data/Cleaned/New_cleaning/dev.json'
    assert_str = "make sure to set PATH for cisco_dev data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert os.path.exists(cisco_dev.PATH), \
                        cisco_dev.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(cisco_dev, self).__init__(cisco_dev.PATH, **kwargs)


class cisco_test(json_dataset):
    """
    dataset for cisco_test with arguments configured for convenience

    command line usage: `--train-data cisco_test`
    """
    PATH = '/raid/antoloui/Master-thesis/Data/Cleaned/New_cleaning/test.json'
    assert_str = "make sure to set PATH for cisco_test data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert os.path.exists(cisco_test.PATH), \
                        cisco_test.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(cisco_test, self).__init__(cisco_test.PATH, **kwargs)
        
        
class wikipedia(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = 'data/wikipedia/wikidump_lines.json'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert os.path.exists(wikipedia.PATH), \
                        wikipedia.assert_str
        if not kwargs:
            kwargs = {}
            kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)


class webtext(json_dataset):
    """
    dataset for webtext with arguments configured for convenience

    command line usage: `--train-data webtext`
    """
    PATH = 'data/webtext/data.json'
    assert_str = "make sure to set PATH for webtext data_utils/corpora.py"
    def __init__(self, **kwargs):
        assert os.path.exists(webtext.PATH), \
                        webtext.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(webtext, self).__init__(webtext.PATH, **kwargs)


NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'webtext': webtext,
    'cisco_train': cisco_train,
    'cisco_dev': cisco_dev,
    'cisco_test': cisco_test,
}

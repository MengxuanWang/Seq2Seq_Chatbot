import re
import os
import codecs
import collections
import cPickle

import numpy as np

PAD = '_PAD'
GO =  '_GO'
EOS = '_EOS'
UNK = '_UNK'
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

class TextLoader():
    def __init__(self, data_dir, encoding='utf8'):
        self.data_dir = data_dir
        self.encoding = encoding

        input_file = os.path.join(data_dir, "dgk_shooter_min.conv")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        source_file = os.path.join(data_dir, "source.npy")
        target_file = os.path.join(data_dir, "target.npy")

        if not (os.path.exists(vocab_file)):
            print "reading text file ... "
            self.preprocess(input_file, vocab_file, source_file, target_file)
        else:
            print "loading preprocessd files ... "
            self.load_preprocessed(vocab_file, source_file, target_file)

    def preprocess(self, input_file, vocab_file, source_file, target_file):
        with codecs.open(input_file, "r", encoding='utf8') as f:
            data = f.read()
        lines = re.split(r'\n', data)
        counter = collections.Counter()
        sources, targets = [], []

        preline = ''
        for line in lines:
            line = re.sub(r'\s+', '', line)
            if line.startswith('E'): # split
                preline = ''
                continue
            elif line.startswith('M'):
                line = line[1:].split('/')
                line = line[:-1]
                counter.update(collections.Counter(line))
                if line and preline:
                    sources.append(preline)
                    targets.append(line)
                preline = line

        counter_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*counter_pairs)
        self.chars = START_VOCAB + list(self.chars)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(self.vocab_size)))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        tensor_sources = [list(map(self.vocab.get, source))
                          for source in sources]
        tensor_targets = [list(map(self.vocab.get, target))
                          for target in targets]
        self.tensor_sources = np.array(tensor_sources)
        self.tensor_targets = np.array(tensor_targets)
        np.save(source_file, self.tensor_sources)
        np.save(target_file, self.tensor_targets)

    def load_preprocessed(self, vocab_file, source_file, target_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(self.vocab_size)))
        self.tensor_sources = np.load(source_file)
        self.tensor_targets = np.load(target_file)

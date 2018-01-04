#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
use this class for encoding sparse data
'''

from six import integer_types
from collections import defaultdict
#import numpy
class sparse(object):
    'the encoding class for maxent'
    def __init__(self, labels, mapping, unseen_features=False,
                 alwayson_features=False):
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError('Mapping values must be exactly the '
                             'set of integers from 0...len(mapping)')

        self._labels = list(labels)
        """A list of attested labels."""

        self._mapping = mapping
        """dict mapping from (fname,fval,label) -> fid"""

        self._length = len(mapping)
        """The length of generated joint feature vectors."""

        self._alwayson = None
        """dict mapping from label -> fid"""

        self._unseen = None
        """dict mapping from fname -> fid"""

        if alwayson_features:
            self._alwayson = dict((label, i+self._length)
                                  for (i, label) in enumerate(labels))
            self._length += len(self._alwayson)

        if unseen_features:
            fnames = set(fname for (fname, fval, label) in mapping)
            self._unseen = dict((fname, i+self._length)
                                for (i, fname) in enumerate(fnames))
            self._length += len(fnames)

    def encode(self, featureset, label):
        # Inherit docs.
        encoding = []

        # Convert input-features to joint-features:
        for fname, fval in featureset.items():
            # Known feature name & value:
            if (fname, fval, label) in self._mapping:
                encoding.append((self._mapping[fname, fval, label], 1))

            # Otherwise, we might want to fire an "unseen-value feature".
            elif self._unseen:
                # Have we seen this fname/fval combination with any label?
                for label2 in self._labels:
                    if (fname, fval, label2) in self._mapping:
                        break # we've seen this fname/fval combo
                # We haven't -- fire the unseen-value feature
                else:
                    if fname in self._unseen:
                        encoding.append((self._unseen[fname], 1))

        # Add always-on features:
        if self._alwayson and label in self._alwayson:
            encoding.append((self._alwayson[label], 1))

        return encoding


    def describe(self, f_id):
        # Inherit docs.
        if not isinstance(f_id, integer_types):
            raise TypeError('describe() expected an int')
        try:
            self._inv_mapping
        except AttributeError:
            self._inv_mapping = [-1]*len(self._mapping)
            for (info, i) in self._mapping.items():
                self._inv_mapping[i] = info

        if f_id < len(self._mapping):
            (fname, fval, label) = self._inv_mapping[f_id]
            return '%s==%r and label is %r' % (fname, fval, label)
        elif self._alwayson and f_id in self._alwayson.values():
            for (label, f_id2) in self._alwayson.items():
                if f_id == f_id2:
                    return 'label is %r' % label
        elif self._unseen and f_id in self._unseen.values():
            for (fname, f_id2) in self._unseen.items():
                if f_id == f_id2:
                    return '%s is unseen' % fname
        else:
            raise ValueError('Bad feature id')


    def labels(self):
        # Inherit docs.
        return self._labels


    def length(self):
        # Inherit docs.
        return self._length


    @classmethod
    def store(cls, train_toks, count_cutoff=0, labels=None, **options):
        mapping = {}              # maps (fname, fval, label) -> fid
        seen_labels = set()       # The set of labels we've encountered
        count = defaultdict(int)  # maps (fname, fval) -> count

        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError('Unexpected label %s' % label)
            seen_labels.add(label)

            # Record each of the features.
            for (fname, fval) in tok.items():

                # If a count cutoff is given, then only add a joint
                # feature once the corresponding (fname, fval, label)
                # tuple exceeds that cutoff.
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)

        if labels is None:
            labels = seen_labels
        return cls(labels, mapping, **options)
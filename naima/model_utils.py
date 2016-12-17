# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import hashlib
from astropy import units as u
import warnings


def memoize(func):
    """ Cache decorator for functions inside model classes """

    def model(cls, energy, *args, **kwargs):
        try:
            memoize = cls._memoize
            cache = cls._cache
            queue = cls._queue
        except AttributeError:
            memoize = False

        if memoize:
            # Allow for dicts or tables with energy column, Quantity array or
            # Quantity scalar
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        'ignore',
                        getattr(np, 'VisibleDeprecationWarning', None))
                    energy = u.Quantity(energy['energy'])
            except (TypeError, ValueError, IndexError):
                pass

            try:
                # tostring is 10 times faster than str(array).encode()
                bstr = energy.value.tostring()
            except AttributeError:
                # scalar Quantity
                bstr = str(energy.value).encode()

            data = [hashlib.sha256(bstr).hexdigest()]

            data.append(energy.unit.to_string())
            data.append(str(kwargs.get('distance', 0)))

            if args:
                data.append(str(args))
            if hasattr(cls, 'particle_distribution'):
                models = [cls, cls.particle_distribution]
            else:
                models = [cls]
            for model in models:
                if hasattr(model, 'param_names'):
                    for par in model.param_names:
                        data.append(str(getattr(model, par)))

            token = u''.join(data)
            digest = hashlib.sha256(token.encode()).hexdigest()

            if digest in cache:
                return cache[digest]

        result = func(cls, energy, *args, **kwargs)

        if memoize:
            # remove first item in queue and remove from cache
            if len(queue) > 16:
                key = queue.pop(0)
                cache.pop(key, None)
            # save last result to cache
            queue.append(digest)
            cache[digest] = result

        return result

    model.__name__ = func.__name__
    model.__doc__ = func.__doc__

    return model

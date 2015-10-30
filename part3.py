from collections import OrderedDict
import functools


class LRUCache(OrderedDict):
    def __init__(self, size, *args, **kwargs):
        """Least Recently Used cache with a maximum size of `size`

        Example usage:
            >>> c = LRUCache(2)
            >>> c[1] = 1
            >>> c[2] = 2
            >>> 1 in c
            True
            >>> c[3] = 3
            >>> 1 in c
            False
        """
        self.size = size
        OrderedDict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        # update access time
        try:
            self.move_to_end(key)
        except KeyError:
            pass
        return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value):
        # update access time
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

        # enforce cache size by removing the least-recently used key
        if len(self) > self.size:
            self.popitem(last=False)


def memoize(k):
    """Wraps a unary function f with an LRU-cache of size k."""

    def inner(f):

        cache = LRUCache(k)

        @functools.wraps(f)
        def wrapper(x):
            if x not in cache:
                cache[x] = f(x)
            return cache[x]

        return wrapper

    return inner


if __name__ == '__main__':
    import doctest
    doctest.testmod()

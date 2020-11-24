import numpy as np

from base.base import Space


class MultiDiscrete(Space):
    """ Search space with multiple discrete choices. """
    """ Example: MultiDiscrete([1,2,3,4,5])"""

    def __init__(self, ns):  # pylint: disable=invalid-name
        ns = np.array(ns)
        if ns.ndim != 1 or ns.dtype not in {np.dtype('int32'), np.dtype('int64')}:
            raise ValueError("ns must be a single dimension array of convertable "f"to int32 or int64, got {ns}")
        self.ns = ns  # pylint: disable=invalid-name

    @property
    def maxchoices(self):
        return max(self.ns)

    def __len__(self):
        return len(self.ns)

    def contains(self, element):
        element = np.asarray(element)
        return (element.ndim == 1
                and element.dtype in {np.dtype('int32'), np.dtype('int64')}
                and np.all((0 <= element) & (element < self.ns)))

    def sample(self):
        return np.array([np.random.randint(n) for n in self.ns])


class SkipSpace(MultiDiscrete):
    """ Space representing n skip connections. """

    def __init__(self, n, value):
        super().__init__([value] * n)


class ChooseSpace(Space):
    """ Space for picking one from multiple spaces. """
    """ Example: ChooseSpace.from_collections(([1],[2])) - this will define a ChooseSpace, to choose between two MultiDiscrete spaces"""
    """          ChooseSpace.sample() - will return a random sample from ChooseSpace (will choose some random space from the overall 
                 space of spaces) - int"""

    def __init__(self, spaces):
        self.spaces = spaces

    @classmethod
    def from_collection(cls, spaces):
        """ Creates an instance given a nested collection where each element
        represents a MultiDiscrete space.
        """
        return cls([MultiDiscrete(s) for s in spaces])

    def __len__(self):
        return 1

    @property
    def maxchoices(self):
        return max(len(self.spaces), max(s.maxchoices for s in self.spaces))

    def contains(self, element):
        element = np.asarray(element)
        return (element.ndim == 1
                and element.dtype in {np.dtype('int32'), np.dtype('int64')}
                and np.all(element < len(self.spaces)))

    def sample(self):
        return [np.random.randint(len(self.spaces))]


class SearchSpace(Space):
    """ Search space consisting of a collection of search spaces. """
    """ Example: SearchSpace.from_collection([([2], [3]), [4]]) - [ChooseSpace (two MultiDiscrete spaces), MultiDiscrete space]"""
    """          SearchSpace.sample() - will return a random sample from all of the subspaces of the search space"""

    def __init__(self, spaces):
        self.spaces = spaces

    @classmethod
    def from_collection(cls, spaces, add_skip_connections=True):
        """ Creates instance from nested collections. """

        def make_space(elem):
            return (ChooseSpace.from_collection(elem) if isinstance(next(iter(elem)), (list, tuple, np.ndarray))
                    else MultiDiscrete(elem))

        return cls(list(chain.from_iterable((make_space(elem), SkipSpace(n, 2)) if add_skip_connections and n
                                            else (make_space(elem),) for n, elem in enumerate(spaces))))

    def __len__(self):
        if not any(isinstance(s, ChooseSpace) for s in self.spaces):
            return sum(map(len, self.spaces))
        return None  # don't have a well-defined length

    @property
    def maxchoices(self):
        return max(space.maxchoices for space in self.spaces)

    def contains(self, element):
        element = np.asarray(element)
        i = 0
        for space in self.spaces:
            if isinstance(space, ChooseSpace):
                if not space.contains(element[i]):
                    return False
                space = space.spaces[element[i]]
                i += 1
            if not space.contains(element[i:i + len(space.ns)]):
                return False
            i += len(space.ns)
        return True

    def sample(self):
        sample = []
        for space in self.spaces:
            if isinstance(space, ChooseSpace):
                i = space.sample()
                sample.append(i)
                space = space.spaces[i]
            sample.extend(space.sample())
        return sample

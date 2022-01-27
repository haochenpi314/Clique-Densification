import numpy as np
import bisect

# numpy array that supports temporal slicing
# https://numpy.org/devdocs/user/basics.subclassing.html


class TArray(np.ndarray):

    def __new__(cls, data, time):
        obj = np.asarray(data).view(cls)
        timestamps = np.asarray(time)
        is_sorted = np.all(timestamps[:-1] <= timestamps[1:])
        obj.timestamps = timestamps
        if not is_sorted:
            order = np.argsort(timestamps)
            obj = obj.view(np.ndarray)[order].view(cls)
            obj.timestamps = timestamps[order]
            # obj.timestamps = np.sort(timestamps)

        obj = obj.view(cls)
        return obj

    def __getitem__(self, timestamp):
        if isinstance(timestamp, slice):
            return self.__timeslice(timestamp)
        else:
            return self.view(np.ndarray)[timestamp]

    def __timeslice(self, timestamp):
        # excludes ending elements
        timestamps = self.timestamps
        start = None
        stop = None
        if timestamp.start is not None:
            start = bisect.bisect_left(timestamps, timestamp.start)
        if timestamp.stop is not None:
            stop = bisect.bisect_left(timestamps, timestamp.stop)
        data = self.view(np.ndarray)[start:stop:None]
        timestamps = self.timestamps[start:stop:None]
        return TArray(data, timestamps)

    def index_to_time(self, index):
        timestamps = self.timestamps
        new_index = bisect.bisect_right(timestamps, timestamps[index])
        timestamp = None
        if new_index < len(timestamps):
            timestamp = timestamps[new_index]
        return timestamp

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamps = getattr(obj, 'timestamps', None)

    def __reduce__(self):
        pickled_state = super(TArray, self).__reduce__()
        new_state = pickled_state[2] + (self.timestamps,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.timestamps = state[-1]
        super(TArray, self).__setstate__(state[0:-1])

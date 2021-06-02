import os
import pickle
import queue
import re
import typing
import time
import multiprocessing as mp

import numpy as np
from gwpy.timeseries import TimeSeriesDict
from stillwater import StreamingInferenceProcess, Package


def _get_file_gps_timestamp(fname):
    # subtract one from file creation time to
    # account for the latency incurred by waiting
    # for the file to be created
    file_creation_timestamp = os.stat(fname).st_ctime - 1
    return file_creation_timestamp


class GwfFrameFileDataSource(StreamingInferenceProcess):
    def __init__(
        self,
        input_pattern: str,
        channels: typing.List[str],
        kernel_stride: float,
        sample_rate: float,
        preproc_file: str,
        name: typing.Optional[str] = None
    ):
        self.input_pattern = input_pattern
        self.channels = channels
        self.kernel_stride = kernel_stride
        self.sample_rate = sample_rate
        self._update_size = int(kernel_stride * sample_rate)

        self._idx = None
        self._data = None
        self._self_q = mp.Queue()

        super().__init__(name=name)

    def __iter__(self):
        self.start()
        return self

    def _get_initial_timestamp(self):
        input_dir, input_pattern = os.path.split(self.input_pattern)
        prefix, postfix = input_pattern.split("{}")
        regex = re.compile(
            "(?<={})[0-9]{}(?={})".format(prefix, "{10}", postfix)
        )
        timestamps = map(regex.search, os.listdir(input_dir))
        if not any(timestamps):
            raise ValueError(
                "Couldn't find any timestamps matching the "
                f"pattern {self.input_pattern}"
            )
        timestamps = [int(t.group(0)) for t in timestamps if t is not None]
        return max(timestamps)

    def _get_data(self):
        if self._idx is None:
            self._idx = 0
            self._t0 = self._get_initial_timestamp()

        start = self._idx * self._update_size
        stop = (self._idx + 1) * self._update_size

        if self._data is None or stop > self._data.shape[1]:
            # try to load in the next second's worth of data
            # if it takes more than a second to get created,
            # then assume the worst and raise an error
            start_time = time.time()
            path = self.path_pattern.format(self._t0)
            while time.time() - start_time < 3:
                try:
                    data = TimeSeriesDict.read(path, self.channels)
                    break
                except FileNotFoundError:
                    time.sleep(1e-3)
            else:
                raise ValueError(f"Couldn't find next timestep file {path}")
            self._latency_t0 = _get_file_gps_timestamp(path)

            # resample the data and turn it into a numpy array
            data.resample(self.sample_rate)
            data = np.stack(
                [data[channel].value for channel in self.channels]
            ).astype("float32")

            self.children["writer"].send((path, data[0], self._latency_t0))
            data = data[1:]

            if self._data is not None and start < self._data.shape[1]:
                leftover = self._data[:, start:]
                data = np.concatenate([leftover, data], axis=1)

            self._data = data
            self._t0 += 1
            self._idx = 0

        # return the next piece of data
        x = self._data[:, start:stop]

        # offset the frame's initial time by the time
        # corresponding to the first sample of stream
        self._idx += 1
        t0 = self._latency_t0 + self._idx * self.kernel_stride
        return Package(x=x, t0=t0)

    def _do_stuff_with_data(self, package):
        self._self_q.put(package)

    def __next__(self):
        while True:
            try:
                return self._self_q.get_nowait()
            except queue.Empty:
                time.sleep(1e-6)


class GwfFrameFileWriter(StreamingInferenceProcess):
    def __init__(
        self,
        output_dir,
        channel_name,
        sample_rate,
        kernel_stride,
        name=None
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.channel_name = channel_name
        self.sample_rate = sample_rate
        self.kernel_stride = kernel_stride

        self._strains = []
        self._noise = np.array([])
        super().__init__(name=name)

    def _get_data(self):
        if self.parents["reader"].poll():
            stuff = self.parents["reader"].recv()
            self._strains.append(stuff)

        if self.parents["client"].poll():
            return self.parents["client"].recv()
        return None

    def _do_stuff_with_data(self, package):
        self._noise = self._noise.append(package.x[0, :kernel_stride])

        if len(self._noise) >= self.sample_rate:
            noise, self._noise = np.split(self._noise, self.sample_rate)
            fname, strain, t0 = self._strains.pop(0)

            strain = strain

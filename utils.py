import glob
import logging
import os
import pickle
import queue
import re
import time
import typing
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.signal as sig
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from stillwater import ExceptionWrapper, StreamingInferenceProcess, Package


def _get_file_timestamp(fname):
    # subtract one from file creation time to
    # account for the latency incurred by waiting
    # for the file to be created
    file_creation_timestamp = os.stat(fname).st_ctime
    return file_creation_timestamp


def _load_preproc(fname: str) -> typing.Tuple[float, float]:
    with open(fname, "wb") as f:
        preproc = pickle.load(f)
        mean = preproc["mean"]
        std = preproc["std"]
    return mean, std


class AsyncGwfReader(StreamingInferenceProcess):
    def __init__(
        self,
        input_pattern: str,
        channels: typing.List[str],
        sample_rate: float,
        inference_sampling_rate: float,
        preproc_file: str,
        N: typing.Optional[int] = None,
        name: typing.Optional[str] = None
    ):
        self.input_pattern = input_pattern
        self.channels = channels
        self.sample_rate = sample_rate
        self.update_size = sample_rate // inference_sampling_rate

        self._idx = 0
        self._data = None
        self._t0 = None
        self._n = 0
        self._N = N
        self._last_time = time.time()

        self._self_q = mp.Queue()
        self._writer_q = mp.Queue()

        self.mean, self.std = _load_preproc(preproc_file)
        super().__init__(name=name)

    def _get_initial_timestamp(self):
        # find all the gwf frames in the input dir
        input_dir = os.path.dirname(self.input_pattern)
        fs = glob.glob(os.path.join(input_dir, "*.gwf"))
        if len(fs) == 0:
            raise ValueError(f"No gwf frames in input directory {input_dir}")

        # assume any consecutive 10 integers in the
        # filename are the timestamp
        regex = re.compile("(?<=-)[0-9]{10}(?=-)")
        timestamps = map(regex.search, map(os.path.basename, fs))
        if not any(timestamps):
            raise ValueError(
                "Couldn't find any valid timestamps in "
                f"input directory {input_dir}"
            )

        timestamps = [int(t.group(0)) for t in timestamps if t is not None]
        return max(timestamps)

    def _load_frame(self):
        # try to load in the next second's worth of data
        # timeout after 3 seconds if nothing becomes available
        if self._t0 is None:
            self._t0 = self._get_initial_timestamp()
        if self._N is not None and self._n == self._N:
            raise StopIteration

        start_time = time.time()
        frame_path = self.input_pattern.format(self._t0)
        while (time.time() - start_time) < 0.5:
            try:
                # with redirect_stderr(StringIO()), sys_pipes():
                data = TimeSeriesDict.read(frame_path, self.channels)
                logging.info(f"Loaded {self._n}th frame {frame_path}")
                break
            except FileNotFoundError:
                logging.info(
                    f"No file found for timestamp {self._t0}, waiting..."
                )
                time.sleep(1e-3)
        else:
            raise ValueError(f"Couldn't find next timestep file {frame_path}")

        # resample the data and turn it into a numpy array
        data.resample(self.sample_rate)
        data = np.stack(
            [data[channel].value for channel in self.channels]
        ).astype("float32")
        data = (data - self.mean) / self.std

        # send strain and filename to writer process
        # for subtraction/writing later.
        # On first iteration, send Nones to indicate not
        # to write anything since those estimates will be
        # based on 0-initialized states
        strain, data = data[0], data[1:]
        self._writer_q.put((frame_path, strain))
        self._t0 += 1
        self._n += 1

        return data


class StreamingGwfFrameFileDataSource(AsyncGwfReader):
    def _get_data(self):
        start = self._idx * self.update_size
        stop = (self._idx + 1) * self.update_size

        # if we haven't loaded a frame or don't have enough
        # data left in the current frame to generate a state
        # update, load in the next frame and reset slice idx
        if self._data is None or stop >= self._data.shape[1]:
            data = super()._load_frame()

            # if we have leftover data from the last frame,
            # append it to the start of this frame
            if self._data is not None and start < self._data.shape[1]:
                leftover = self._data[:, start:]
                data = np.concatenate([leftover, data], axis=1)

            # reset everything
            self._data = data
            self._idx = 0
            start, stop = 0, self.update_size

        # return the next piece of data
        x = self._data[:, start:stop]

        # offset the frame's initial time by the time
        # corresponding to the first sample of stream
        self._idx += 1
        sleep_time = self.update_size / (2 * self.sample_rate)
        while (time.time() - self._last_time) < (sleep_time):
            time.sleep(1e-6)
        self._last_time = time.time()
        return Package(x=x, t0=self._last_time)

    def _break_glass(self, exception):
        super()._break_glass(exception)
        self._self_q.put(ExceptionWrapper(exception))

    def _do_stuff_with_data(self, package):
        self._self_q.put(package)

    def __iter__(self):
        return self

    def __next__(self):
        timeout = 3
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                package = self._self_q.get_nowait()
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
                return package
            except queue.Empty:
                time.sleep(1e-4)
        else:
            raise RuntimeError("No data!")


class GwfFrameFileDataSource(AsyncGwfReader):
    def __init__(
        self,
        input_pattern: str,
        channels: typing.List[str],
        kernel_size: float,
        sample_rate: float,
        inference_sampling_rate: float,
        preproc_file: str,
        name: typing.Optional[str] = None
    ):
        self.kernel_size = kernel_size
        super().__init__(
            input_pattern=input_pattern,
            channels=channels,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            preproc_file=preproc_file,
            name=name
        )

    def _get_data(self):
        data = super()._load_frame()
        self._self_q.put(data)

    def _get_frame_from_q(self):
        start_time = time.time()
        timeout = 3
        while (time.time() - start_time) < timeout:
            try:
                frame = self._self_q.get_nowait()
                break
            except queue.Empty:
                time.sleep(1e-3)
        else:
            raise RuntimeError(f"No frames received for {timeout} seconds")

        if isinstance(frame, ExceptionWrapper):
            frame.reraise()
        return frame

    def __next__(self):
        start = self._idx * self.update_size
        stop = start + self.kernel_size

        if self._data is None:
            # we've never loaded a frame, so load one now
            self._data = self._load_frame()
        elif stop >= self._data.shape[1]:
            # we don't have enough data to build
            # the next frame, so try to get the
            # next frame from our separate process
            frame = self._get_frame_from_q()
            self._data = np.append(self._data, frame, axis=1)

        if start > self.kernel_size:
            # periodically slough off some stale data
            self._data = self._data[:, self.kernel_size:]
            start -= self.kernel_size
            stop -= self.kernel_size
            self._idx = 0

        # return the next piece of data
        x = self._data[:, start:stop]

        # offset the frame's initial time by the time
        # corresponding to the first sample of stream
        self._idx += 1
        sleep_time = self.kernel_stride / (2 * self.sample_rate)
        while (time.time() - self._last_time) < (sleep_time):
            time.sleep(1e-6)
        self._last_time = time.time()
        return Package(x=x[None], t0=self._last_time)

    def _break_glass(self, exception):
        self._self_q.put(ExceptionWrapper(exception))

    def __iter__(self):
        return self


class GwfFrameFileWriter(StreamingInferenceProcess):
    def __init__(
        self,
        output_dir: str,
        channel_name: str,
        sample_rate: float,
        q: queue.Queue,
        preproc_file: str,
        name=None
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.channel_name = channel_name
        self.sample_rate = sample_rate
        self.q = q

        self._strains = []
        self._noise = np.array([])
        self._first_strain = True
        self._first_noise = True

        self.mean, self.std = _load_preproc(preproc_file)

        # TODO: generalize
        with open(preproc_file, "wb") as f:
            preproc = pickle.load(f)
            filt_kwargs = {}
            filt_kwargs = {
                i: preproc[f"filt_{i}"] for i in ["fl", "fh", "order"]
            }
            self.filter = partial(self.bandpass, **filt_kwargs)
        super().__init__(name=name)

    @staticmethod
    def bandpass(data, fs, fl, fh, order=None, axis=-1):
        """Copied from deepclean_prod for now

        Parameters
        ----------
        data: array
        fs: sampling frequency
        fl, fh: low and high frequency for bandpass
        axis: axis to apply the filter on

        Returns:
        --------
        data_filt: filtered array
        """
        if order is None:
            order = 8

        # Make filter
        nyq = fs / 2.
        low, high = fl / nyq, fh / nyq   # normalize frequency
        z, p, k = sig.butter(
            order, [low, high], btype='bandpass', output='zpk'
        )
        sos = sig.zpk2sos(z, p, k)

        # Apply filter and return output
        data_filt = sig.sosfiltfilt(sos, data, axis=axis)
        return data_filt

    def _try_recv_and_check(self, conn):
        if conn.poll():
            package = conn.recv()
            if isinstance(package, ExceptionWrapper):
                package.reraise()
            return package

    def _get_data(self):
        try:
            stuff = self.q.get_nowait()
        except queue.Empty:
            pass
        else:
            if not self._first_strain:
                self._strains.append(stuff)
            self._first_strain = False

        package = self._try_recv_and_check(self._parents.client)
        if not self._first_noise:
            return package
        self._first_noise = False

    def _do_stuff_with_data(self, package):
        # add the new inferences to the
        # running noise estimate array
        self._noise = np.append(self._noise, package["output_0"].x[0])

        # TODO: this is 1-second specific, make more general
        if len(self._noise) >= self.sample_rate:
            # if we've accumulated a frame's worth of
            # noise, split it off and subtract it from
            # its corresponding strain
            noise, self._noise = np.split(self._noise, [self.sample_rate])

            frame_path, strain = self._strains.pop(0)
            if frame_path is None:
                # don't write the first frame's worth of data
                # since those estimates will be bad from being
                # streamed on top of the 0 initialized state
                return

            # get the frame timestamp from the filename
            _, fname = os.path.split(frame_path)
            t0 = int(re.search("(?<=-)[0-9]{10}(?=-)", fname).group(0))

            # subtract the noise estimate from the strain
            # and create a gwpy timeseries from it
            noise = noise * self.std + self.mean
            noise = self.filter(noise)
            cleaned = strain - noise
            timeseries = TimeSeries(
                cleaned,
                t0=t0,
                sample_rate=self.sample_rate,
                channel=self.channel_name
            )

            # add "_cleaned" to the filename and write the cleaned
            # strain to this new file in the desired location
            write_fname = os.path.join(self.output_dir, fname)
            timeseries.write(write_fname)

            # let the main process know that we wrote a file and
            # what the corresponding latency to that write was
            latency = time.time() - _get_file_timestamp(frame_path)
            self._children.output.send((write_fname, latency))

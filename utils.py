import glob
import os
import queue
import re
import typing
import time
import multiprocessing as mp
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
import tritonclient.grpc as triton
from google.protobuf import text_format
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from stillwater import ExceptionWrapper, StreamingInferenceProcess, Package
from tritonclient.utils import InferenceServerException
from wurlitzer import sys_pipes


def _get_file_timestamp(fname):
    # subtract one from file creation time to
    # account for the latency incurred by waiting
    # for the file to be created
    file_creation_timestamp = os.stat(fname).st_ctime
    return file_creation_timestamp


def _get_initial_timestamp(input_pattern):
    # find all the gwf frames in the input dir
    input_dir = os.path.dirname(input_pattern)
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


class StreamingGwfFrameFileDataSource(StreamingInferenceProcess):
    def __init__(
        self,
        input_pattern: str,
        channels: typing.List[str],
        update_size: int,
        sample_rate: float,
        preproc_file: str,
        name: typing.Optional[str] = None
    ):
        self.input_pattern = input_pattern
        self.channels = channels
        self.sample_rate = sample_rate
        self.update_size = update_size

        self._idx = None
        self._data = None
        self._last_time = time.time()
        self._self_q = mp.Queue()
        self._writer_q = mp.Queue()

        super().__init__(name=name)

    def _load_frame(self, start):
        # try to load in the next second's worth of data
        # timeout after 3 seconds if nothing becomes available
        start_time = time.time()
        frame_path = self.input_pattern.format(self._t0)
        while time.time() - start_time < 3:
            try:
                with redirect_stderr(StringIO()), sys_pipes():
                    data = TimeSeriesDict.read(frame_path, self.channels)
                break
            except FileNotFoundError:
                time.sleep(1e-3)
        else:
            raise ValueError(f"Couldn't find next timestep file {frame_path}")

        # resample the data and turn it into a numpy array
        data.resample(self.sample_rate)
        data = np.stack(
            [data[channel].value for channel in self.channels]
        ).astype("float32")

        # send strain and filename to writer process
        # for subtraction/writing later.
        # On first iteration, send Nones to indicate not
        # to write anything since those estimates will be
        # based on 0-initialized states
        strain, data = data[0], data[1:]
        self._writer_q.put((frame_path, strain))

        # if we have leftover data from the last frame,
        # append it to the start of this frame
        if self._data is not None and start < self._data.shape[1]:
            leftover = self._data[:, start:]
            data = np.concatenate([leftover, data], axis=1)

        # reset everything
        self._data = data
        self._t0 += 1
        self._idx = 0
        return 0, self.update_size

    def _get_data(self):
        # if we've never loaded a frame, initialize
        # some parameters, using the latest timestamp
        # in the indicated directory to begin
        if self._idx is None:
            self._idx = 0
            self._t0 = self._get_initial_timestamp(self.input_pattern)

        start = self._idx * self.update_size
        stop = (self._idx + 1) * self.update_size

        # if we haven't loaded a frame or don't have enough
        # data left in the current frame to generate a state
        # update, load in the next frame and reset slice idx
        if self._data is None or stop >= self._data.shape[1]:
            start, stop = self._load_frame(start)

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
        while True:
            try:
                package = self._self_q.get_nowait()
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
                return package
            except queue.Empty:
                time.sleep(1e-6)


class GwfFrameFileDataSource(StreamingInferenceProcess):
    def __init__(
        self,
        input_pattern: str,
        channels: typing.List[str],
        kernel_size: float,
        kernel_stride: float,
        sample_rate: float,
        preproc_file: str,
        name: typing.Optional[str] = None
    ):
        self.input_pattern = input_pattern
        self.channels = channels
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

        self._idx = 0
        self._data = None
        self._t0 = None

        self._last_time = time.time()
        self._self_q = mp.Queue()
        self._writer_q = mp.Queue()

        super().__init__(name=name)

    def _get_data(self):
        # try to load in the next second's worth of data
        # timeout after 3 seconds if nothing becomes available
        if self._t0 is None:
            self._t0 = _get_initial_timestamp(self.input_pattern)

        start_time = time.time()
        frame_path = self.input_pattern.format(self._t0)
        while time.time() - start_time < 3:
            try:
                with redirect_stderr(StringIO()), sys_pipes():
                    data = TimeSeriesDict.read(frame_path, self.channels)
                break
            except FileNotFoundError:
                time.sleep(1e-3)
        else:
            raise ValueError(f"Couldn't find next timestep file {frame_path}")

        # resample the data and turn it into a numpy array
        data.resample(self.sample_rate)
        data = np.stack(
            [data[channel].value for channel in self.channels]
        ).astype("float32")

        # send strain and filename to writer process
        # for subtraction/writing later.
        # On first iteration, send Nones to indicate not
        # to write anything since those estimates will be
        # based on 0-initialized states
        strain, data = data[0], data[1:]
        self._writer_q.put((frame_path, strain))
        self._self_q.put(data)
        self._t0 += 1

    def _load_frame(self):
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
        # if we've never loaded a frame, initialize
        # some parameters, using the latest timestamp
        # in the indicated directory to begin
        start = self._idx * self.kernel_stride
        stop = start + self.kernel_size

        # if we haven't loaded a frame or don't have enough
        # data left in the current frame to generate a state
        # update, load in the next frame and reset slice idx
        if self._data is None:
            self._data = self._load_frame()
        elif stop >= self._data.shape[1]:
            frame = self._load_frame()
            self._data = np.append(self._data, frame, axis=1)

        if start > self.kernel_size:
            self._data = self._data[:, self.kernel_size:]
            start -= self.kernel_size
            stop -= self.kernel_size
            self._idx = 0

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


class GwfFrameFileWriter(StreamingInferenceProcess):
    def __init__(
        self,
        output_dir,
        channel_name,
        sample_rate,
        q,
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
        super().__init__(name=name)

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
            cleaned = strain - noise
            timeseries = TimeSeries(
                cleaned,
                t0=t0,
                sample_rate=self.sample_rate,
                channel=self.channel_name
            )

            # add "_cleaned" to the filename and write the cleaned
            # strain to this new file in the desired location
            write_fname = fname.replace(".gwf", "_cleaned.gwf")
            write_fname = os.path.join(self.output_dir, write_fname)
            timeseries.write(write_fname)

            # let the main process know that we wrote a file and
            # what the corresponding latency to that write was
            latency = time.time() - _get_file_timestamp(frame_path)
            self._children.output.send((write_fname, latency))


class DummyClient(StreamingInferenceProcess):
    def __init__(self, reader, name=None):
        self.reader = iter(reader)
        super().__init__(name=name)

    def _get_data(self):
        return next(self.reader)

    def _do_stuff_with_data(self, package):
        package.x = package.x.sum(axis=0) * 0.
        self._children.writer.send(package)

    def __exit__(self, *exc_args):
        super().__exit__(*exc_args)
        self.reader.try_elegant_stop()


class ModelController:
    def __init__(self, url, model_repo):
        self.url = url
        self.model_repo = model_repo

    @property
    def client(self):
        return triton.InferenceServerClient(self.url)

    def deepclean_config(self, output_size):
        postfix = f"output-size={output_size}"
        return os.path.join(
            self.model_repo, f"deepclean_{postfix}", "config.pbtxt"
        )

    def scale(self, output_size, gpus, count):
        model_config = triton.model_config_pb2.ModelConfig()
        with open(self.deepclean_config(output_size), "r") as f:
            text_format.Merge(f.read(), model_config)

        instance_group = triton.model_config_pb2.ModelInstanceGroup(
            count=count, gpus=gpus
        )
        try:
            model_config.instance_group[0].MergeFrom(instance_group)
        except IndexError:
            model_config.instance_group.append(instance_group)

        with open(self.deepclean_config(output_size), "w") as f:
            f.write(str(model_config))

    def load(self, output_size):
        model_name = f"dc-stream_output_size={output_size}"
        for i in range(2):
            try:
                self.client.load_model(model_name)
                break
            except InferenceServerException as e:
                exc = str(e)
                time.sleep(5 * (1 - i))
        else:
            raise RuntimeError(exc)

    def unload(self, output_size):
        model_name = f"dc-stream_kernel-stride={output_size}"
        for i in range(2):
            try:
                self.client.unload_model(model_name, unload_dependents=True)
                break
            except InferenceServerException as e:
                exc = str(e)
                time.sleep(5 * (1 - i))
        else:
            raise RuntimeError(exc)

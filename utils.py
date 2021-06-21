import concurrent.futures
import glob
import os
import queue
import re
import threading
import time
import typing
import multiprocessing as mp
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
import requests
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


class GenericGwfFrameFileDataSource(StreamingInferenceProcess):
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

        self._idx = 0
        self._data = None
        self._t0 = None
        self._last_time = time.time()

        self._self_q = mp.Queue()
        self._writer_q = mp.Queue()

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
        self._t0 += 1

        return data


class StreamingGwfFrameFileDataSource(GenericGwfFrameFileDataSource):
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
            self._t0 += 1
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
        while True:
            try:
                package = self._self_q.get_nowait()
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
                return package
            except queue.Empty:
                time.sleep(1e-6)


class GwfFrameFileDataSource(GenericGwfFrameFileDataSource):
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
        self.kernel_size = kernel_size
        super().__init__(
            input_pattern=input_pattern,
            channels=channels,
            update_size=kernel_stride,
            sample_rate=sample_rate,
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


_hexes = "[0-9a-f]"
_gpu_id_pattern = "-".join([_hexes + f"{{{i}}}" for i in [8, 4, 4, 4, 12]])
_res = [
    re.compile('(?<=nv_inference_)[a-z_]+(?=_duration_us{gpu_uuid="GPU-)'),
    re.compile(f'(?<=gpu_uuid="GPU-){_gpu_id_pattern}(?=")'),
    re.compile(f'(?<=model=")[a-z_]+(?=",version=)'),
    re.compile("(?<=} )[0-9.]+$")
]


class ServerMonitor(mp.Process):
    def __init__(self, ips, filename):
        self.ips = ips
        self.filename = filename

        self.header = (
            "ip,step,gpu_id,model,process,time (us),interval,count,utilization"
        )
        self._last_times = {}
        self._counts = {}
        self._times = {}

        self._stop_event = mp.Event()
        self._error_q = mp.Queue()
        super().__init__()

    def _get_data_for_ip(self, ip, step):
        response = requests.get(f"http://{ip}:8002/metrics")
        response.raise_for_status()

        request_time = time.time()
        try:
            last_time = self._last_times[ip]
            interval = request_time - last_time
        except KeyError:
            pass
        finally:
            self._last_times[ip] = request_time

        data, counts, utilizations = "", {}, {}
        models_to_update = []
        rows = response.content.decode().split("\n")

        # start by collecting the number of new inference
        # counts and the GPU utilization
        for row in rows:
            if row.startswith("nv_inference_exec_count"):
                try:
                    gpu_id, model, value = [
                        r.search(row).group(0) for r in _res[1:]
                    ]
                except AttributeError:
                    continue

                if model in _MODELS:
                    value = int(float(value))
                    try:
                        count = value - self._counts[(ip, gpu_id, model)]
                        if count > 0:
                            # if no new inferences were registered, we
                            # won't need to update this model below
                            models_to_update.append((gpu_id, model))
                        counts[(ip, gpu_id, model)] = count
                    except KeyError:
                        # we haven't recorded this model before, so
                        # there's no need to update it
                        pass
                    finally:
                        # add or update the number of inference counts
                        # for this model on this GPU
                        self._counts[(ip, gpu_id, model)] = value

            elif row.startswith("nv_gpu_utilization"):
                try:
                    gpu_id, value = [r.search(row).group(0) for r in _res[1::2]]
                except AttributeError:
                    continue
                utilizations[gpu_id] = value

        for row in rows:
            try:
                process, gpu_id, model, value = [
                    r.search(row).group(0) for r in _res
                ]
            except AttributeError:
                continue

            value = float(value)
            index = (ip, process, gpu_id, model)

            if (gpu_id, model) not in models_to_update:
                # we don't need to record a row of data for this
                # GPU/model combination, either because this is
                # the first loop or because we didn't record any
                # new inferences during this interval
                if model in _MODELS and index not in self._times:
                    # we don't have a duration for this process
                    # on this node/GPU/model combo, so create one
                    self._times[index] = value
                continue

            delta = value - self._times[index]
            self._times[index] = value
            utilization = utilizations[gpu_id]
            count = counts[(ip, gpu_id, model)]

            data += "\n" + ",".join([
                ip,
                str(step),
                gpu_id,
                model,
                process,
                str(delta),
                str(interval),
                str(count),
                utilization
            ])
        return data

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def check(self):
        try:
            e = self._error_q.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Error in monitor: " + e)

    def run(self):
        f = open(self.filename, "w")
        f.write(self.header)

        lock = threading.Lock()

        def target(ip):
            step = 0
            try:
                while not self.stopped:
                    data = self._get_data_for_ip(ip, step)
                    if data:
                        with lock:
                            f.write(data)
                        step += 1
            except Exception:
                self.stop()
                raise

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(target, ip) for ip in self.ips]

        try:
            while len(futures) > 0:
                done, futures = concurrent.futures.wait(futures, timeout=1e-2)
                for future in done:
                    exc = future.exception()
                    if exc is not None:
                        raise exc
        except Exception as e:
            self._error_q.put(str(e))
        finally:
            f.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()
        self.join(1)
        try:
            self.close()
        except ValueError:
            self.terminate()
import logging
import queue
import sys
import time

import numpy as np
import typeo
from stillwater import StreamingInferenceClient
from stillwater.utils import ExceptionWrapper
from utils import GwfFrameFileDataSource, GwfFrameFileWriter, DummyClient


def main(
    url: str,
    model_name: str,
    model_version: int,
    input_pattern: str,
    channels: str,
    kernel_stride: float,
    sample_rate: int,  # TODO: can get from model config,
    output_dir: str
):
    with open(channels, "r") as f:
        channels = f.read().splitlines()
        channels = [x.split()[0] for x in channels[:21]]

    client = StreamingInferenceClient(
        url, model_name, model_version, name="client"
    )
    writer = GwfFrameFileWriter(
        output_dir,
        channel_name=channels[0],
        sample_rate=sample_rate,
        name="writer"
    )

    source = GwfFrameFileDataSource(
        input_pattern,
        channels=channels,
        kernel_stride=kernel_stride,
        sample_rate=sample_rate,
        preproc_file="",
        name="reader"
    )
    writer.add_parent(source)
    # client = DummyClient(source, name="client")
    client.add_data_source(source, child=writer)

    conn_out = writer.add_child("output")
    last_recv_time = time.time()
    latency, n = 0.0, 0
    with client, writer:
        while True:
            fname = None
            if conn_out.poll():
                fname = conn_out.recv()
                last_recv_time = time.time()
            elif (time.time() - last_recv_time) > 3:
                raise RuntimeError(
                    "No files written in last 3 seconds, exiting"
                )

            # make sure our pipeline didn't raise an error
            if isinstance(fname, ExceptionWrapper):
                fname.reraise()
            elif fname is not None:
                fname, e2e_latency = fname

            # update our running latency measurement
            for i in range(20):
                try:
                    _, t0, __, tf = client._metric_q.get_nowait()
                except queue.Empty:
                    break
                n += 1
                latency += (tf - t0 - latency) / n

            # sleep for a second to release the GIL no matter
            # what happened, then continue if no filenames
            # got returned
            time.sleep(0.1)
            if fname is None:
                continue

            logging.info(
                "Wrote cleaned data to {} with {:0.2f} ms of end-to-end "
                "latency, average inf latency {:0.2f} ms".format(
                    fname, e2e_latency * 1000, latency * 1000
                )
            )
            n += 1
            if n == 100:
                break


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    flags = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d - %(levelname)-8s %(message)s",
        stream=sys.stdout,
        datefmt="%H:%M:%S",
        level=logging.INFO
    )
    main(**vars(flags))

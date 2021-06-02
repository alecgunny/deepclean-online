import logging
import sys
import time

import typeo
from stillwater import ThreadedMultiStreamClient
from stillwater.utils import ExceptionWrapper
from utils import GwfFrameFileDataSource, GwfFrameFileWriter


def main(
    url: str,
    model_name: str,
    model_version: int,
    input_pattern: str,
    channels: str,
    sample_rate: int,  # TODO: can get from model config,
    output_pattern: str
):
    with open(channels, "r") as f:
        channels = f.read().splitlines()

    client = ThreadedMultiStreamClient(
        url, model_name, model_version, name="client"
    )
    writer = GwfFrameFileWriter(output_pattern, channels[0], name="writer")

    source = GwfFrameFileDataSource(
        input_pattern, channels, sample_rate, name="reader"
    )
    writer.add_parent(source)
    client.add_source(source, child=writer)

    conn_out = writer.add_child("output")
    client.start()
    last_recv_time = time.time()
    # latency, n = 0.0, 0
    with client:
        while True:
            fname = None
            if conn_out.poll():
                fname = conn_out.recv()
            elif (time.time() - last_recv_time) > 3:
                raise RuntimeError(
                    "No files written in last 3 seconds, exiting"
                )

            # make sure our pipeline didn't raise an error
            if isinstance(fname, ExceptionWrapper):
                fname.reraise()

            # # update our running latency measurement
            # for i in range(20):
            #     try:
            #         _, t0, __, tf = client._metric_q.get_nowait()
            #     except queue.Empty:
            #         break
            #     n += 1
            #     latency += (tf - t0 - latency) / n

            # sleep for a second to release the GIL no matter
            # what happened, then continue if no filenames
            # got returned
            time.sleep(0.1)
            if fname is None:
                continue

            logging.info(
                "Wrote cleaned data to {}, average latency {:0.2}f ms".format(
                    fname, latency * 1000
                )
            )


if __name__ == "__main__":
    parser = typeo.make_parser(main)

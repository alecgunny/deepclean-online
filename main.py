import logging
import queue
import sys
import time

import typeo
from stillwater import StreamingInferenceClient
from stillwater.utils import ExceptionWrapper
from utils import (
    GwfFrameFileDataSource,
    GwfFrameFileWriter,
    StreamingGwfFrameFileDataSource
)


def main(
    url: str,
    model_name: str,
    model_version: int,
    input_pattern: str,
    channels: str,
    output_size: int,
    sample_rate: int,  # TODO: can get from model config,
    output_dir: str,
    stats_file: str,
    N: int = 100,
    stream: bool = False
):
    with open(channels, "r") as f:
        channels = f.read().splitlines()
        channels = [x.split()[0] for x in channels[:22]]

    client = StreamingInferenceClient(
        url, model_name, model_version, name="client"
    )

    if stream:
        source = StreamingGwfFrameFileDataSource(
            input_pattern,
            channels=channels,
            update_size=output_size,
            sample_rate=sample_rate,
            preproc_file="",
            name="reader"
        )
    else:
        source = GwfFrameFileDataSource(
            input_pattern,
            channels=channels,
            kernel_size=sample_rate,
            kernel_stride=output_size,
            sample_rate=sample_rate,
            preproc_file="",
            name="reader"
        )

    writer = GwfFrameFileWriter(
        output_dir,
        channel_name=channels[0],
        sample_rate=sample_rate,
        q=source._writer_q,
        name="writer"
    )

    client.add_data_source(source, child=writer)
    conn_out = writer.add_child("output")

    last_recv_time = time.time()
    n = 0
    with source, client, writer:
        while True:
            fname = None
            if conn_out.poll():
                fname = conn_out.recv()
                last_recv_time = time.time()
            elif (time.time() - last_recv_time) > 20:
                raise RuntimeError(
                    "No files written in last 20 seconds, exiting"
                )

            # make sure our pipeline didn't raise an error
            if isinstance(fname, ExceptionWrapper):
                fname.reraise()
            elif fname is not None:
                fname, e2e_latency = fname

            # sleep for a second to release the GIL no matter
            # what happened, then continue if no filenames
            # got returned
            time.sleep(0.1)
            if fname is None:
                continue

            logging.info(
                "Wrote cleaned data to {} with {:0.2f} ms "
                "of end-to-end latency".format(
                    fname, e2e_latency * 1000,
                )
            )
            n += 1
            if n == N:
                break

    _, start_time = client._metric_q.get_nowait()
    with open(stats_file, "a") as f:
        while True:
            try:
                stuff = client._metric_q.get_nowait()
            except queue.Empty:
                break

            _, t0, __, tf = stuff
            t0 -= start_time
            tf -= start_time
            f.write(f"\n{output_size},{t0},{tf}")


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

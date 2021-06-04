import glob
import logging
import os
import sys
import typing

import typeo
from main import main as run_one_expt
from utils import ModelController


def main(
    model_repo: str,
    input_pattern: str,
    kernel_strides: typing.List[float]
):
    url = "0.0.0.0:8001"
    controller = ModelController(url, model_repo)
    for num_gpus in [1, 2]:
        for scale in [1, 2, 4, 6, 8]:
            fname = f"client-stats_gpus={num_gpus}_scale={scale}.csv"
            with open(fname, "w") as f:
                f.write("stride,start,stop")

            logging.info(
                f"Scaling deepclean to gpus={num_gpus} instances={scale}"
            )
            controller.scale([i for i in range(num_gpus)], scale)

            for kernel_stride in kernel_strides:
                model_name = f"dc-stream_kernel-stride={kernel_stride}"
                logging.info(f"Loading model {model_name}")
                controller.load(model_name)

                logging.info("Model loaded, running expt")
                run_one_expt(
                    url,
                    model_name,
                    model_version="1",
                    input_pattern=input_pattern,
                    channels="channels.txt",
                    kernel_stride=kernel_stride,
                    sample_rate=4000,
                    output_dir=".",
                    fname=fname
                )
                logging.info("Removing files")
                for f in glob.glob("*.gwf"):
                    os.remove(f)
                logging.info(f"Unloading model {model_name}")


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

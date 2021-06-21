import glob
import logging
import os
import sys
import time

import typeo
from main import main as run_one_expt
from utils import ModelController


def main(
    model_repo: str,
    input_pattern: str,
    max_output_size: int
):
    url = "0.0.0.0:8001"
    controller = ModelController(url, model_repo)
    for num_gpus in [1, 2, 4]:
        gpus = [i for i in range(num_gpus)]
        for scale in [1, 2, 4, 6, 8]:
            fname = f"client-stats_gpus={num_gpus}_scale={scale}.csv"
            if os.path.exists(fname):
                logging.info(
                    f"Expt for gpus={num_gpus} instances={scale} already exists"
                )
                continue

            with open(fname, "w") as f:
                f.write("size,start,stop")

            logging.info(
                f"Scaling deepclean to gpus={num_gpus} instances={scale}"
            )
            for i in range(4, max_output_size):
                controller.scale(i, gpus, scale)
                model_name = f"dc-stream_output_size={i}"
                logging.info(f"Loading model {model_name}")
                controller.load(i)

                logging.info("Model loaded, running expt")
                for _ in range(5):
                    try:
                        run_one_expt(
                            url,
                            model_name,
                            model_version=1,
                            input_pattern=input_pattern,
                            channels="channels.txt",
                            output_size=i,
                            sample_rate=4000,
                            output_dir=".",
                            stats_file=fname,
                            N=20
                        )
                        break
                    except Exception as e:
                        if "StatusCode" not in str(e):
                            raise
                        logging.warn("Encountered gRPC error, trying again")
                        time.sleep(5)
                    finally:
                        logging.info("Removing files")
                        for f in glob.glob("*.gwf"):
                            os.remove(f)
                else:
                    raise RuntimeError("Too many gRPC errors")

                logging.info(f"Unloading model {model_name}")
                controller.unload(i)


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

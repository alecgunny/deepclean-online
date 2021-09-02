import glob
import logging
import os
import sys
import time

import numpy as np
import typeo
import pandas as pd
from main2 import main as run_one_expt
from utils import ModelController


def main(
    model_repo: str,
    input_pattern: str,
    min_output_size: int,
    max_output_size: int
):
    url = "0.0.0.0:8001"
    controller = ModelController(url, model_repo)
    for num_gpus in [1, 2, 4]:
        gpus = [i for i in range(num_gpus)]
        for scale in [1, 2, 4, 6, 8]:
            fname = f"client-stats_gpus={num_gpus}_scale={scale}.csv"
            min_size = min_output_size
#             if os.path.exists(fname):
#                 df = pd.read_csv(fname)
#                 min_size = df["size"].max() + 1
#                 if min_size >= max_output_size:
#                     logging.info(
#                         f"All expts for gpus={num_gpus} instances={scale} run"
#                     )
#                     continue
#                 logging.info(
#                     f"Starting tests for gpus={num_gpus} instances={scale} "
#                     f"at output size {min_size}"
#                 )
#             else:
#                 min_size = min_output_size
#                 with open(fname, "w") as f:
#                     f.write("size,start,stop")
               

            logging.info(
                f"Scaling deepclean to gpus={num_gpus} instances={scale}"
            )
            for i in range(min_size, max_output_size):
                if os.path.exists(fname.replace("client", f"server-{i}")):
                    logging.info(f"Skipping expt gpus={gpus} scale={scale} size={i}")
                    continue
                controller.scale(i, gpus, scale)
                model_name = f"dc-stream_output_size={i}"
                logging.info(f"Loading model {model_name}")
                controller.load(i)

                logging.info("Model loaded, running expt")
                for _ in range(10):
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
                            N=20,
                            stream=True
                        )
                        break
                      #   df = pd.read_csv(f"server-{i}-stats_gpus={num_gpus}_scale={scale}.csv")
                      #   throughput = np.percentile(df["count"] / df["interval"], 50)
                      #   latency = np.percentile(df["us"] / df["count"], 50)
                      #   if (throughput < (0.9 * 4000 / i)) or (latency > 10000):
                      #       if throughput < (0.9 * 4000 / i):
                      #           warning = f"Median throughput {throughput:0.1f} too low"
                      #       elif latency > 10000:
                      #           warning = f"Median latency {latency:0.1f} too high"
                      #       logging.warning(warning)
                      #       time.sleep(1)
                      #   else:
                      #       break
                    except Exception as e:
                        if "StatusCode" in str(e):
                            logging.warning(f"Encountered gRPC error {e}, trying again")
                            time.sleep(10)
                        elif isinstance(e, IndexError):
                            logging.warning(f"Encountered IndexError, trying again")
                            time.sleep(1)
                        else:
                            raise
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

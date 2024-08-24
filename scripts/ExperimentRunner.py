from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
from RRTstar_3D_with_RI_optimization import run_experiment

if __name__ == "__main__":
    ### Configure this variables to run an experiment

    NUM_WORKERS = 10  # Number of workers (experiments) to run in parallel
    NUM_TRIALS_PER_WORKER = 10
    ROBOT_TYPE = "ur5" # choose between ur5 or kuka
    RM_GAIN = 1e-3 # Positive -> Maximization, Negative -> Minimization
    VERBOSE = False

    dictionary_l = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(run_experiment, exp_id, NUM_TRIALS_PER_WORKER, ROBOT_TYPE, RM_GAIN, VERBOSE)
            for exp_id in range(NUM_WORKERS)
        ]
        for future in futures:
            res = future.result()
            dictionary_l.append(res)
            exp_id = res["name"]
            computation_time = res["time"]
            end_time = time.time()
            print("time required: ", end_time - start_time)
            np.savez(
                f"experiment_{exp_id}",
                q=res["q"],
                traj_points=res["traj_points"],
                rm_masses=res["rm_masses"],
                time=computation_time,
            )

    import ipdb

    ipdb.set_trace()

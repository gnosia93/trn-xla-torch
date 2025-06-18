#!/usr/bin/env python

# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""
This script perform extraction of graphs (HLO protobuf files) and do
parallel compilation.
During extraction, you will see multiple messages:
'Extracting graphs for ahead-of-time parallel compilation. No compilation was done.'
Furthermore, you will see fake execution so the loss results are random.
"""

from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
import argparse
import psutil
import os
import shutil
import sys
import concurrent.futures
import collections
import json

import time
import math
import traceback
import codecs

import torch_neuronx.parallel_compile.analyze_utils as analyze
from libneuronxla.libncc import setup_args
from libneuronxla.neuron_cc_cache import create_compile_cache, CacheUrl, CACHE_STRUCTURE_INFO
import libneuronxla.neuron_cc_wrapper
from libneuronxla import logger

# Need to save it in /tmp/ so that nodes in a cluster do not overwrite
# each other
NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE = os.environ.get(
    "NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE", None)
NEURONCC_PARALLEL_COMPILE_ENV_INFO = """
    NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE: By default the utility will skip the compilation if
     there is an error in training script. You can set the NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE
     to continue compilation of the accumulated collected graphs."""
NEURON_PARALLEL_COMPILE_DUMP_RESULTS = os.environ.get(
    "NEURON_PARALLEL_COMPILE_DUMP_RESULTS", None)


LOGGER = logger.get_logger("NEURON_PARALLEL_COMPILE")


class StatusReport:

    def __init__(self) -> None:
        self.report = {}
        self.status_counter = collections.defaultdict(int)
        self.total = 0
        self.start_timer()

    def start_timer(self) -> None:
        self.start_time = time.time()

    def update(self, target, succeed, retry, compile_time):
        if target not in self.report.keys():
            self.status_counter[succeed] += 1
            self.total += 1
        self.report[target] = {"status": succeed, "retry": retry, "compile_time": compile_time}

    def __str__(self) -> str:
        results_str = json.dumps({
            "compilation_summary": self.status_counter,
            "compilation_report": self.report,
            "start_time": self.start_time,
            "compilation_time": time.time() - self.start_time
        }, indent=4)
        if NEURON_PARALLEL_COMPILE_DUMP_RESULTS:
            # the result filename has time encoded since in multi-node setup each node
            # would dump the file and we want to retain the compile times.
            with open(f"neuron_parallel_compile_results_{time.time()}.json", "w") as f:
                f.write(results_str)
        return results_str

    def log_summary(self):
        LOGGER.info(f"Total graphs: {self.total}")
        LOGGER.info(
            f"Total successful compilations: {self.status_counter[True]}")
        LOGGER.info(f"Total failed compilations: {self.status_counter[False]}")


def _get_default_parallel_worker_count():
    # keeping about 64GB for each of the compiler process.
    # For trn1, this comes to 8processes per worker, and for trn2 this comes to 32
    return math.ceil(psutil.virtual_memory()._asdict()['total'] / (64 * 1024 * 1024 * 1024))


def compile_task_helper(compiled_hlo_status, compile_cache, task_list, workdir, dump=None, compile_work_dir=None):
    counter = 0
    for hlo_to_compile in list(task_list):
        cache_entry = compile_cache.try_lookup_for_compile(hlo_to_compile)
        # We'll get None if the HLO is already done, or locked by another worker
        if cache_entry is None:
            continue
        with cache_entry:
            # need to check again, since lock is released after NEFF is generated
            if cache_entry.exists:
                continue
            start_time = time.time()
            try:
                status, retry = libneuronxla.neuron_cc_wrapper.compile_cache_entry(
                    os.path.join(workdir, f"model.{cache_entry.key}.neff"),
                    cache_entry, dump=dump, work_dir=compile_work_dir)
                compiled_hlo_status[hlo_to_compile] = (status, retry, time.time() - start_time)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    LOGGER.error(f"Process {os.getpid()} failed compilation: {str(e)}")
                else:
                    LOGGER.error(f"Subprocess {os.getpid()} encountered error: {str(e)}")
                compiled_hlo_status[hlo_to_compile] = (False, 0, time.time() - start_time)
        # report the status
        counter += 1
        if counter % 10 == 0:
            hlos_rem, *_ = compile_cache.get_hlos()
            if len(hlos_rem) == 0:
                break


def get_task_range(tasks, worker_id, worker_num):
    """
    statically distribute the task based on workder_id and worker_num
    """

    size = math.floor(len(tasks) / worker_num)
    start = worker_id * size
    end = min((worker_id + 1) * size, len(tasks))

    leftovers = len(tasks) - size * worker_num
    leftover_task = []
    if leftovers > 0 and worker_id + size * worker_num < len(tasks):
        leftover_task = [tasks[worker_id + size * worker_num]]

    return tasks[start: end] + leftover_task


def get_k_task_range(tasks, worker_id, worker_num, k):

    neighbor_tasks = []

    for i in range(1, k + 1):
        w_id = worker_id - i if worker_id - i >= 0 else worker_num + worker_id - i
        neighbor_tasks += get_task_range(tasks, w_id, worker_num)

    return neighbor_tasks


def compile_task(workdir, cache_url, hlos, global_worker_id, global_worker_num,
                 static_schedule=False, dump=None, compile_work_dir=None):
    """
    If worker_num is not None, assign a tasklist to each worker evenly based on worker_num.
    If worker finish ealier, it will try to steal from it previous 8 workers.
    """

    compile_cache = create_compile_cache(cache_url)

    compiled_hlo_status = {}

    if static_schedule:
        # do assigned task range
        LOGGER.info(f"worker {global_worker_id} start its assigned tasks....")
        tasklist = get_task_range(hlos, global_worker_id, global_worker_num)
        LOGGER.info(f"worker {global_worker_id} get {len(tasklist)} tasks "
                    + f"(num_hlos {len(hlos)}, worker_num {global_worker_num})")

        compile_task_helper(compiled_hlo_status, compile_cache, tasklist, workdir, dump=dump,
                            compile_work_dir=compile_work_dir)

        # start to steal neighbor's jobs after finishing assigned tasks
        # if the num is not evenly divisible by worker_num, the last worker tends to do less job but start to steal
        tasklist = get_k_task_range(hlos, global_worker_id, global_worker_num, 8)
        LOGGER.info(f"worker {global_worker_id} start to steal ({len(tasklist)}) from others tasks....")
        compile_task_helper(compiled_hlo_status, compile_cache, tasklist, workdir, dump=dump,
                            compile_work_dir=compile_work_dir)
    else:
        LOGGER.info(f"worker {global_worker_id} starts dynamic scheduleing on {len(hlos)}....")
        compile_task_helper(compiled_hlo_status, compile_cache, hlos, workdir, dump=dump,
                            compile_work_dir=compile_work_dir)

    LOGGER.info(f"worker {global_worker_id} finished with num of tasks {len(compiled_hlo_status)}....")
    # report the status in the end
    compile_cache.get_hlos()
    return compiled_hlo_status


def get_hlos_from_run_log(trial_run_log):
    # New graphs are detected by specific message matching key
    hlo_key = "Extracting graphs"
    new_hlo_list = []
    with codecs.open(trial_run_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines():
            # Move temporary MODULE_* files into workdir before checking if there are any
            # new graphs. In try_compilations, compile only new graphs (those without
            # corresponding neffs).
            if hlo_key in line:
                model_path = line.split("Extracting graphs (")[1].split(")")[0]
                new_hlo_list.append(model_path)

    format_str = '\n\t'
    LOGGER.info(f"New graph list from script {len(new_hlo_list)}: {format_str.join(new_hlo_list)}")
    return new_hlo_list


def parallel_compile(work_dir, cache_url, num_parallel=8, hlo_tasklist=None, node_id=0, world_size=None,
                     static_schedule=False, dump=None, compile_work_dir=None):
    total_npc_time_start = time.time()
    status_report = StatusReport()
    compile_cache = create_compile_cache(cache_url)

    if static_schedule and hlo_tasklist is not None:
        raise RuntimeError("Static schedule is not supported when hlo_tasklist is provided (compile-and-run)")

    if world_size is None and static_schedule:
        raise RuntimeError("Cannot use static_schedule when hlo_list is provided or world_size is None")

    # get todo hlos as tasklist if not provided
    if hlo_tasklist is None:
        hlos, *_ = compile_cache.get_hlos()
        hlo_tasklist = list(hlos)

    # get all hlos if static_schedule is on
    if static_schedule:
        all_hlos = set()
        hlos_sets = compile_cache.get_hlos()
        for hlos_set in hlos_sets:
            all_hlos = all_hlos.union(hlos_set)
        hlo_tasklist = list(all_hlos)

    # check if there are remain tasks
    if len(hlo_tasklist) == 0:
        LOGGER.info("No todo hlos found in cache or hlo_tasklist, parallel_compile return before spawning workers....")
        return

    hlo_tasklist = list(sorted(hlo_tasklist))
    LOGGER.debug(f"get hlo_tasklist: {[(i, hlo) for i, hlo in enumerate(hlo_tasklist)]}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel) as executor:
        features = [executor.submit(compile_task, work_dir, cache_url, hlo_tasklist,
                                    global_worker_id=num_parallel * node_id + i,
                                    global_worker_num=world_size * num_parallel if world_size else None,
                                    static_schedule=static_schedule,
                                    dump=dump, compile_work_dir=compile_work_dir) for i in range(num_parallel)]
        concurrent.futures.wait(features)
        for i, feature in enumerate(features):
            try:
                compiled_hlo_status = feature.result()
                for compiled_hlo, (status, retry, compile_time) in compiled_hlo_status.items():
                    status_report.update(compiled_hlo, status, retry, compile_time)
            except Exception as e:
                LOGGER.info(f"sub-process {i} got exception: {str(e)}")

    LOGGER.info(status_report)
    status_report.log_summary()
    total_npc_time_end = time.time()
    if NEURON_PARALLEL_COMPILE_DUMP_RESULTS:
        total_results = json.dumps({
            "total_neuron_parallel_compile_time": total_npc_time_end - total_npc_time_start
        }, indent=4)
        with open(f"total_neuron_parallel_compile_results_{time.time()}.json", "w") as f:
            f.write(total_results)
    return status_report.status_counter[False]


def _create_workdir(workdir, remove_exists=True):
    if os.path.exists(workdir) and remove_exists:
        LOGGER.info("Removing existing workdir {}".format(workdir))
        shutil.rmtree(workdir)
    Path(workdir).mkdir(parents=True, exist_ok=True)


def _get_train_cmd(args, unparsed_args, cache_path_log):
    if args.command == "analyze":
        workdir, cmd = analyze.get_training_script_args(
            args.analyze_verbosity, unparsed_args)
        _create_workdir(workdir)
    else:
        cmd = unparsed_args
    return cmd


def run_train_script(args, unparsed_args, cache_path_log, trial_run_log):
    if not unparsed_args:
        args.print_help()
        sys.exit(0)

    LOGGER.info((
        "Running trial run (add option to terminate trial run early;"
        " also ignore trial run's generated outputs, i.e. loss, checkpoints)"))

    os.environ["NEURON_EXTRACT_GRAPHS_ONLY"] = "1"
    os.environ["NEURON_PARALLEL_COMPILE_CACHE_PATH_LOG"] = cache_path_log
    if args.command == "analyze":
        os.environ["NEURON_ANALYZE_MODEL"] = "1"
        os.environ["NEURON_ANALYZE_ARTIFACTS_PATH"] = analyze.ANALYZE_ARTIFACTS_PATH
        os.environ["XLA_HLO_DEBUG"] = "1"

    graph_tracing_env = os.environ.copy()
    graph_tracing_env["PATH"] = f"/usr/sbin:/sbin:{graph_tracing_env['PATH']}"

    cmd = _get_train_cmd(args, unparsed_args, cache_path_log)
    torchrun_redirect = 'torchrun' in cmd and any('--redirect' in arg for arg in cmd)
    if torchrun_redirect:
        LOGGER.error("neuron_parallel_compile is currently incompatible with torchrun redirect. \
            Please use --collect and then --compile if you want to use the --redirect flag.")
        sys.exit(1)
    LOGGER.info(f"Running cmd: {cmd}")
    try:
        with Popen(cmd, stdout=PIPE, stderr=STDOUT, env=graph_tracing_env) as process, open(
            trial_run_log, "bw"
        ) as file:
            while True:
                byte = process.stdout.read(1)
                if byte:
                    sys.stdout.buffer.write(byte)
                    sys.stdout.flush()
                    file.write(byte)
                else:
                    break
        return_code = process.returncode
    except BaseException as e:
        return_code = -1
        LOGGER.error(e)
    if return_code != 0:
        LOGGER.error("There was an error in the training script.")
        if NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE:
            LOGGER.info((
                "NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE is set, continuing"
                " to compile"))
            return
        sys.exit(1)


def _extract_args():
    parser = argparse.ArgumentParser(
        prog="neuron_parallel_compile",
        description="neuron_parallel_compile is an utility to extract graphs from trial "
        "run of your script, perform parallel compilation of the graphs, and "
        "populate the persistent cache with compiled graphs. Your trial run "
        "should be limited to about 100 steps, enough for the utility to extract"
        " the different graphs needed for full execution.\n"
        "To avoid hang during extraction, please make sure to use xm.save instead "
        "of torch.save to save checkpoints.\n"
        "After parallel compile, the actual run of your script will be faster since"
        " the compiled graphs are already cached. There may be additional compilations"
        " due to unreached execution paths, or changes in parameters such as number"
        " of data parallel workers.\n\n"
        "Envionment Variables:"
        f"\t {libneuronxla.neuron_cc_wrapper.NEURONCC_WRAPPER_ENV_INFO}\n"
        f"\t {NEURONCC_PARALLEL_COMPILE_ENV_INFO}\n"
        f"Cache Structure:"
        f"\t {CACHE_STRUCTURE_INFO}\n",
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--command",
        choices=[
            "collect-and-compile",
            "collect",
            "clear-locks",
            "clear-neffs",
            "compile",
            "clean",
            "scan",
            "scan-with-failed",
            "analyze"],
        default="collect-and-compile",
        help=(
            "collect-and-compile (default): Collect traced graphs of a short run and compile on collected graphs.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment\n"
            "         or --cache_dir option in NEURON_CC_FLAGS to select cache location.\n"
            "collect: Run a short run (i.e. 10 steps) to collect traced graphs of training loops.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "compile: Compile on collected graphs.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "clear-locks: Clear lock files in cache. WARNING: Please ensure there are no other running compilation"
            "         tasks under same cache.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "clear-nefffs: Clear NEFF file for debug. WARNING: Please ensure there are no other running compilation"
            "         tasks under same cache.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "clean: Clean cached artifacts. WARNING: this command removes the cache dir.\n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "scan: Scan the cache dir to list todo/done/locked hlos. (Note failed hlos will be included in done,"
            "         use scan-with-failed otherwise) \n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "scan-with-failed: Scan the cache dir to list todo/done/locked/failed hlos. \n"
            "         Use NEURON_COMPILE_CACHE_URL in environment to select cache location.\n"
            "analyze: Analyze graphs to determine operator support for a short run. \n"

        )
    )

    parser.add_argument(
        "--num_parallel",
        default=_get_default_parallel_worker_count(),
        type=int,
        help="default: 8 for trn1 & 32 for trn2"
    )
    parser.add_argument(
        "--node_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--static_schedule",
        action='store_true',
        help="When static_schedule is on and world_size/node_id is provided, it will statically divided up the \n"
        "entire cache dir into task regions to be assign to each worker, if worker finishes its task regions, \n"
        "it will try to steal task from its neighbors. Note that it will scan the entire cache to include all \n"
        "hlos (todo, locked, done). So only recommended for large scale compilation with a clean cache \n"
        "(doesn't have much done tasks and share with tohers) \n"
    )
    parser.add_argument(
        "--analyze-output",
        type=str,
        default=analyze.DEFAULT_ANALYZE_OUTPUT_DIR,
        help=analyze.ANALYZE_OUTPUT_HELP
    )
    parser.add_argument(
        "--analyze-verbosity",
        type=str,
        choices=["1", "2"],
        default="2",
        help=analyze.ANALYZE_VERBOSITY_HELP,
    )
    parser.add_argument(
        "--parallel_compile_workdir",
        help="Folder for hlos in this compile",
        default=f"/tmp/{os.getenv('USER', 'no-user')}/parallel_compile_workdir")

    parser.add_argument(
        "--log_level",
        choices=[
            "INFO",
            "DEBUG",
            "ERROR",
            "WARNING"],
        default="INFO")

    args, unparsed_args = parser.parse_known_args()
    args.print_help = parser.print_help

    return args, unparsed_args


def main():

    args, unknown_args = _extract_args()

    pid = os.getpid()
    cache_path_log = "{}/cache_path_pid{}_log.txt".format(
        args.parallel_compile_workdir, pid)
    trial_run_log = "{}/trial_run_pid{}_log.txt".format(
        args.parallel_compile_workdir, pid)

    ncc_args, _, _ = setup_args()
    if ncc_args.no_cache:
        LOGGER.WARNING("--no_cache will be ignored")

    cache_dir = None
    if ncc_args.cache_dir:
        cache_dir = ncc_args.cache_dir

    _create_workdir(args.parallel_compile_workdir)
    ret = 0
    if args.command == "collect":
        run_train_script(args, unknown_args, cache_path_log, trial_run_log)
    elif args.command == "clean":
        cache_url = CacheUrl.get_cache_url(cache_dir)
        compile_cache = create_compile_cache(cache_url)
        compile_cache.clean()
        LOGGER.info("Cleanup done.")
    elif args.command == "clear-locks":
        cache_url = CacheUrl.get_cache_url(cache_dir)
        compile_cache = create_compile_cache(cache_url)
        compile_cache.clear_locks()
        LOGGER.info("Cleanup locks done.")
    elif args.command == "clear-neffs":
        cache_url = CacheUrl.get_cache_url()
        compile_cache = create_compile_cache(cache_url)
        compile_cache.clear_neffs()
        LOGGER.info("Cleanup neffs done.")
    elif args.command == "scan" or args.command == "scan-with-failed":
        cache_url = CacheUrl.get_cache_url(cache_dir)
        compile_cache = create_compile_cache(cache_url)
        filtered_hlos, locked, done, failed = compile_cache.get_hlos()
        LOGGER.info(
            f"cache scan for {cache_url.url}: \n \ttodo[{len(filtered_hlos)}]: {filtered_hlos}\n \
            \tlocked[{len(locked)}]: {locked}\n \tdone[{len(done)}]: {done}\n \tfailed[{len(failed)}]: {failed}\n")
    elif args.command == "collect-and-compile":
        run_train_script(args, unknown_args, cache_path_log, trial_run_log)
        # Use the cache path from NeuronCache (passed through env or
        # --cache_dir)
        try:
            with open(cache_path_log, 'r') as f:
                cache_path = f.read()
        except BaseException:
            cache_path = None

        new_hlo_list = get_hlos_from_run_log(trial_run_log)
        cache_url = CacheUrl.get_cache_url(cache_path)

        ret = parallel_compile(args.parallel_compile_workdir, cache_url, args.num_parallel, hlo_tasklist=new_hlo_list,
                               world_size=args.world_size, node_id=args.node_id, static_schedule=args.static_schedule,
                               dump=ncc_args.dump, compile_work_dir=ncc_args.compile_workdir)
    elif args.command == "analyze":
        run_train_script(args, unknown_args, cache_path_log, trial_run_log)
        collected_hlos = get_hlos_from_run_log(trial_run_log)
        analyze.analyze_model(args, collected_hlos)
    else:
        assert (args.command == "compile")
        cache_url = CacheUrl.get_cache_url(cache_dir)
        ret = parallel_compile(args.parallel_compile_workdir, cache_url, args.num_parallel,
                               world_size=args.world_size, node_id=args.node_id, static_schedule=args.static_schedule,
                               dump=ncc_args.dump, compile_work_dir=ncc_args.compile_workdir)

    sys.exit(ret)


if __name__ == "__main__":
    main()

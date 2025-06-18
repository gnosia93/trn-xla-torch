
```
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 parallel_compile]$ pwd
/opt/aws_neuronx_venv_pytorch_2_6/lib/python3.10/site-packages/torch_neuronx/parallel_compile

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 parallel_compile]$ ls -la
total 48
drwxrwxr-x.  3 ec2-user ec2-user    83 Jun 18 04:41 .
drwxrwxr-x. 16 ec2-user ec2-user 16384 May 20 18:10 ..
drwxrwxr-x.  2 ec2-user ec2-user    90 May 20 18:10 __pycache__
-rw-rw-r--.  1 ec2-user ec2-user  7776 May 20 18:10 analyze_utils.py
-rw-rw-r--.  1 ec2-user ec2-user 22547 May 20 18:10 neuron_parallel_compile.py
```

```
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
```

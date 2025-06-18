
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
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 libneuronxla]$ pwd
/opt/aws_neuronx_venv_pytorch_2_6/lib/python3.10/site-packages/libneuronxla
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 libneuronxla]$ ls -la
total 196288
drwxrwxr-x.   4 ec2-user ec2-user     16384 May 20 18:10 .
drwxr-xr-x. 519 ec2-user root         32768 Jun 18 04:17 ..
-rw-rw-r--.   1 ec2-user ec2-user      9806 May 20 18:10 LICENSE.txt
-rw-rw-r--.   1 ec2-user ec2-user    219180 May 20 18:10 THIRD-PARTY-LICENSES.txt
-rw-rw-r--.   1 ec2-user ec2-user      1795 May 20 18:10 __init__.py
drwxrwxr-x.   2 ec2-user ec2-user     16384 May 20 18:10 __pycache__
-rw-rw-r--.   1 ec2-user ec2-user       939 May 20 18:10 analyze_interface.py
-rw-rw-r--.   1 ec2-user ec2-user       919 May 20 18:10 hook.py
-rw-rw-r--.   1 ec2-user ec2-user      7531 May 20 18:10 libncc.py
-rwxrwxr-x.   1 ec2-user ec2-user 200595072 May 20 18:10 libneuronpjrt.so
-rw-rw-r--.   1 ec2-user ec2-user       309 May 20 18:10 libneuronpjrt_path.py
-rw-rw-r--.   1 ec2-user ec2-user      1597 May 20 18:10 libnrt.py
-rw-rw-r--.   1 ec2-user ec2-user       508 May 20 18:10 logger.py
-rw-rw-r--.   1 ec2-user ec2-user     21272 May 20 18:10 neuron_cc_cache.py
-rw-rw-r--.   1 ec2-user ec2-user     17424 May 20 18:10 neuron_cc_wrapper.py
-rw-rw-r--.   1 ec2-user ec2-user       808 May 20 18:10 profiler.py
drwxrwxr-x.   3 ec2-user ec2-user       106 May 20 18:10 proto
-rw-rw-r--.   1 ec2-user ec2-user      3220 May 20 18:10 version.py
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

```
def call_neuron_compiler(work_dir, input_file, compile_flags,
                         output_file, execution_mode=ExecutionMode.LAZY,
                         framework="XLA", dump=None):
    cmd = ["neuronx-cc",
           "compile",
           f"--framework={framework}",
           input_file,
           "--output",
           output_file, ] + compile_flags

    with tempfile.TemporaryDirectory() as tmpdir:
        if dump is not None:
            tmpdir = os.path.abspath(dump)
            tmpdir = os.path.join(
                tmpdir, f'pid{os.getpid()}-program{GlobalCounter()()}')
            os.makedirs(tmpdir, exist_ok=True)
            cmd.extend(['--pipeline', 'compile', 'SaveTemps'])
            ver_cmd = ['neuronx-cc', '--version']
            ncc_version = subprocess.check_output(
                ver_cmd, stderr=subprocess.STDOUT).decode()
            ncc_version, *_ = ncc_version.split('\n')
            *_, ncc_version = ncc_version.split('version ')
            with open(os.path.join(tmpdir, 'neuronx_cc_metadata.json'), 'w') as fp:
                json.dump([ncc_version, cmd], fp)
```

```
#!/opt/aws_neuronx_venv_pytorch_2_6/bin/python3
# -*- coding: utf-8 -*-
import re
import sys
from neuronxcc.driver.CommandDriver import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

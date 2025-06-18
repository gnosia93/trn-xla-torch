
#### trn1 인스턴스 생성 ####
* [Neuron DLAMI User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html)
![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/ec2-trn1.png)

#### neuronx 환경 체크 ####
```
[ec2-user@ip-172-31-76-174 ~]$ source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 bin]$ python
Python 3.10.12 (main, May 20 2025, 17:57:16) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.executable
'/opt/aws_neuronx_venv_pytorch_2_6/bin/python'
>>> quit()

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 bin]$ pip show torch_neuronx
Name: torch-neuronx
Version: 2.6.0.2.7.5413+113e6810
Summary: UNKNOWN
Home-page: UNKNOWN
Author:
Author-email:
License: UNKNOWN
Location: /opt/aws_neuronx_venv_pytorch_2_6/lib/python3.10/site-packages
Requires: libneuronxla, numpy, protobuf, psutil, torch, torch-xla
Required-by: neuronx-distributed
```

#### 디바이스 체크 ####
```
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$ lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
    CPU family:           6
    Model:                106
    Thread(s) per core:   2
    Core(s) per socket:   4
    Socket(s):            1
    Stepping:             6
    BogoMIPS:             5799.95
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fx
                          sr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology no
                          nstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse
                          4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_l
                          m abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adj
                          ust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflu
                          shopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd i
                          da arat avx512vbmi pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bita
                          lg tme avx512_vpopcntdq rdpid md_clear flush_l1d arch_capabilities
Virtualization features:  
  Hypervisor vendor:      KVM
  Virtualization type:    full
Caches (sum of all):      
  L1d:                    192 KiB (4 instances)
  L1i:                    128 KiB (4 instances)
  L2:                     5 MiB (4 instances)
  L3:                     54 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-7
Vulnerabilities:          
  Gather data sampling:   Unknown: Dependent on hypervisor status
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Mitigation; Clear CPU buffers; SMT Host state unknown
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW s
                          equence; BHI SW loop, KVM SW loop
  Srbds:                  Not affected
  Tsx async abort:        Not affected

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$ lspci
00:00.0 Host bridge: Intel Corporation 440FX - 82441FX PMC [Natoma]
00:01.0 ISA bridge: Intel Corporation 82371SB PIIX3 ISA [Natoma/Triton II]
00:01.3 Non-VGA unclassified device: Intel Corporation 82371AB/EB/MB PIIX4 ACPI (rev 08)
00:03.0 VGA compatible controller: Amazon.com, Inc. Device 1111
00:04.0 Non-Volatile memory controller: Amazon.com, Inc. NVMe EBS Controller
00:05.0 Ethernet controller: Amazon.com, Inc. Elastic Network Adapter (ENA)
00:1e.0 System peripheral: Amazon.com, Inc. NeuronDevice (Trainium)
00:1f.0 Non-Volatile memory controller: Amazon.com, Inc. NVMe SSD Controller

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$ neuron-
neuron-bench                   neuron-monitor-cloudwatch.py   neuron-profile
neuron-dump.py                 neuron-monitor-device-view.py  neuron-top
neuron-ls                      neuron-monitor-prometheus.py   
neuron-monitor                 neuron-monitor-top.py          

(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$ neuron-ls
instance-type: trn1.2xlarge
instance-id: i-0a07d4c92c2670ba7
+--------+--------+--------+--------------+
| NEURON | NEURON | NEURON |     PCI      |
| DEVICE | CORES  | MEMORY |     BDF      |
+--------+--------+--------+--------------+
| 0      | 2      | 32 GB  | 0000:00:1e.0 |
+--------+--------+--------+--------------+
```


#### vscode 서버 설치 ###

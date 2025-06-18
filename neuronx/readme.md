## [PyTorch Neuron (“torch-neuronx”) Setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/amazon-linux/torch-neuronx-al2023.html#setup-torch-neuronx-al2023) ##


### 1. trn1 인스턴스 생성 ###
![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/ec2-trn1-2.png)

trn1.2xlarge / Amazon Linux 2023 AMI / 512GB 이상 EBS

### 2. Neuron 드라이버 설치 ###
![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/ssh-login.png)
```
# Configure Linux for Neuron repository updates
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

# Update OS packages 
sudo yum update -y

# Install OS headers 
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

# Install git 
sudo yum install git -y

# install Neuron Driver
sudo yum install aws-neuronx-dkms-2.* -y

# Install Neuron Runtime 
sudo yum install aws-neuronx-collectives-2.* -y
sudo yum install aws-neuronx-runtime-lib-2.* -y

# Install Neuron Tools 
sudo yum install aws-neuronx-tools-2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH
```

### 3. EFA 설치 ###
```
# Install EFA Driver (only required for multi-instance training)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz 
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key 
cat aws-efa-installer.key | gpg --fingerprint 
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig 
tar -xvf aws-efa-installer-latest.tar.gz 
cd aws-efa-installer && sudo bash efa_installer.sh --yes 
cd 
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer
```

### 4. torch-neuronx 파이썬 패키지 설치 (pytorch 2.6) ###
```
# Install External Dependency
sudo yum install -y libxcrypt-compat

# Install Python venv 
sudo yum install -y gcc-c++ 

# Create Python venv
python3.9 -m venv aws_neuron_venv_pytorch 

# Activate Python venv 
source aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip 

# Install Jupyter notebook kernel
pip install ipykernel 
python3.9 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository 
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli 
python -m pip install wget 
python -m pip install awscli 

# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx torchvision
```


### 5. neuronx 환경 체크 ###
```
(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 bin]$ python
Python 3.10.12 (main, May 20 2025, 17:57:16) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.executable
'/opt/aws_neuronx_venv_pytorch/bin/python'
>>> quit()

(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 bin]$ pip show torch_neuronx
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

### 6. 디바이스 체크 ###
```
(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 ~]$ lscpu
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

(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 ~]$ lspci
00:00.0 Host bridge: Intel Corporation 440FX - 82441FX PMC [Natoma]
00:01.0 ISA bridge: Intel Corporation 82371SB PIIX3 ISA [Natoma/Triton II]
00:01.3 Non-VGA unclassified device: Intel Corporation 82371AB/EB/MB PIIX4 ACPI (rev 08)
00:03.0 VGA compatible controller: Amazon.com, Inc. Device 1111
00:04.0 Non-Volatile memory controller: Amazon.com, Inc. NVMe EBS Controller
00:05.0 Ethernet controller: Amazon.com, Inc. Elastic Network Adapter (ENA)
00:1e.0 System peripheral: Amazon.com, Inc. NeuronDevice (Trainium)
00:1f.0 Non-Volatile memory controller: Amazon.com, Inc. NVMe SSD Controller

(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 ~]$ neuron-
neuron-bench                   neuron-monitor-cloudwatch.py   neuron-profile
neuron-dump.py                 neuron-monitor-device-view.py  neuron-top
neuron-ls                      neuron-monitor-prometheus.py   
neuron-monitor                 neuron-monitor-top.py          

(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 ~]$ neuron-ls
instance-type: trn1.2xlarge
instance-id: i-0a07d4c92c2670ba7
+--------+--------+--------+--------------+
| NEURON | NEURON | NEURON |     PCI      |
| DEVICE | CORES  | MEMORY |     BDF      |
+--------+--------+--------+--------------+
| 0      | 2      | 32 GB  | 0000:00:1e.0 |
+--------+--------+--------+--------------+
```

### 7. 모니터링 ###
```
(aws_neuronx_venv_pytorch) [ec2-user@ip-172-31-76-174 ~]$ neuron-top
```
![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/neuron-top.png)

### 8. vscode 리모트 서버 설치 ###

Access through VS Code remote server
With Visual Studio Code installed on your local machine, you can use the Remote-SSH command to edit and run files that are stored on a Neuron instance. See the VS Code article for additional details.

Select Remote-SSH: Connect to Host… from the Command Palette (F1, ⇧⌘P)  
Enter in the full connection string from the ssh section above: ssh -i “/path/to/sshkey.pem” ubuntu@instance_ip_address  
VS Code should connect and automatically set up the VS Code server.  
Eventually, you should be prompted for a base directory. You can browse to a directory on the Neuron instance.  
In case you find that some commands seem greyed out in the menus, but the keyboard commands still work (⌘S to save or ^⇧` for terminal), you may need to restart VS Code.


## VSCode ##

Access through VS Code remote server
With Visual Studio Code installed on your local machine, you can use the Remote-SSH command to edit and run files that are stored on a Neuron instance. See the VS Code article for additional details.

Select Remote-SSH: Connect to Host… from the Command Palette (F1, ⇧⌘P)  
Enter in the full connection string from the ssh section above: ssh -i “/path/to/sshkey.pem” ubuntu@instance_ip_address  
VS Code should connect and automatically set up the VS Code server.  
Eventually, you should be prompted for a base directory. You can browse to a directory on the Neuron instance.  
In case you find that some commands seem greyed out in the menus, but the keyboard commands still work (⌘S to save or ^⇧` for terminal), you may need to restart VS Code.  

 
## [XLA](https://openxla.org/xla) ##
XLA (Accelerated Linear Algebra) is an open-source compiler for machine learning. The XLA compiler takes models from popular frameworks such as PyTorch, TensorFlow, and JAX, and optimizes the models for high-performance execution across different hardware platforms including GPUs, CPUs, and ML accelerators.

As a part of the OpenXLA project, XLA is built collaboratively by industry-leading ML hardware and software companies, including Alibaba, Amazon Web Services, AMD, Apple, Arm, Google, Intel, Meta, and NVIDIA.

* [PyTorch/XLA](https://docs.pytorch.org/xla/release/r2.7/index.html)
* [PJRT Plugin to Accelerate Machine Learning](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html)
* [Neuron Kernel Interface (NKI)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)  
a bare-metal language and compiler for directly programming NeuronDevices available on AWS Trn/Inf instances. You can use NKI to develop, optimize and run new operators directly on NeuronCores while making full use of available compute and memory resources. NKI empowers ML developers to self-serve and invent new ways to use the NeuronCore hardware, starting NeuronCores v2 (Trainium1) and beyond.

#### AWS Accelerators ####

* [Setting up AWS Trainium for Hugging Face Transformers](https://www.philschmid.de/setup-aws-trainium)
* [Fine-tune BERT for Text Classification on AWS Trainium](https://huggingface.co/docs/optimum-neuron/tutorials/fine_tune_bert)
* [Amazon Trainium and Inferentia workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/06367dba-1077-4a51-967c-477dbbbb48b1/en-US/inf2-lab/stable-diffusion)


## [Torch Script](https://docs.pytorch.org/docs/main/jit.html) ##

* [딥러닝 모델 배포하기 #01 - MLOps PipeLine과 연산 최적화 / 모델 경량화](https://happy-jihye.github.io/dl/torch-1/)
* [딥러닝 모델 배포하기 #02 - TorchScript & Pytorch JIT](https://happy-jihye.github.io/dl/torch-2/)
* [Introduction to TorchScript](https://docs.pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
* [Loading a TorchScript Model in C++](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)
* [TorchScript and PyTorch JIT | Deep Dive](https://www.youtube.com/watch?v=2awmrMRf0dA)


## vLLM ##

* https://docs.vllm.ai/en/latest/getting_started/quickstart.html#installation


## Additional ##

* [CPU, GPU, and NPU](https://levysoft.medium.com/cpu-gpu-and-npu-understanding-key-differences-and-their-roles-in-artificial-intelligence-2913a24d0747)
* [파이썬 코딩도장](https://dojang.io/course/view.php?id=7)
* [Introduction to KubernetesExecutor and KubernetesPodOperator](https://medium.com/uncanny-recursions/introduction-to-kubernetesexecutor-and-kubernetespodoperator-ae9bb809e3b3)

---

* https://developer.arm.com/documentation/102374/0102/Data-processing---arithmetic-and-logic-operations

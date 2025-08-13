## [XLA](https://openxla.org/xla) ##
XLA (Accelerated Linear Algebra) is an open-source compiler for machine learning. The XLA compiler takes models from popular frameworks such as PyTorch, TensorFlow, and JAX, and optimizes the models for high-performance execution across different hardware platforms including GPUs, CPUs, and ML accelerators.

As a part of the OpenXLA project, XLA is built collaboratively by industry-leading ML hardware and software companies, including Alibaba, Amazon Web Services, AMD, Apple, Arm, Google, Intel, Meta, and NVIDIA.

* [PyTorch/XLA](https://docs.pytorch.org/xla/release/r2.7/index.html)
* [PJRT Plugin to Accelerate Machine Learning](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html)

#### AWS Accelerators ####

* [Setting up AWS Trainium for Hugging Face Transformers](https://www.philschmid.de/setup-aws-trainium)
* [Fine-tune BERT for Text Classification on AWS Trainium](https://huggingface.co/docs/optimum-neuron/tutorials/fine_tune_bert)
* [Amazon Trainium and Inferentia workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/06367dba-1077-4a51-967c-477dbbbb48b1/en-US/inf2-lab/stable-diffusion)

## Neuron ##

* [Neuron Runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/neuron-runtime/nrt-api-guide.html#nrt-api-guide)
Library (libnrt) is the intermediate layer between Application + Framework and Neuron Driver + Neuron Device. It provides a C API for initializing the Neuron hardware, staging models and input data, executing inferences and training iterations on the staged models, and retrieving output data
![](https://github.com/gnosia93/trn-xla-torch/blob/main/neuronx/images/neuron-runtime.png)
* [Neuron Compiler](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/compiler/index.html)
* [Neuron Custom C++ Operators in MLP Training](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/neuron-customops/tutorials/customop-mlp-training.html#neuronx-customop-mlp-tutorial)
* [Deploy Containers with Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/containers/index.html)
* [Neuron Kernel Interface (NKI)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)  
a bare-metal language and compiler for directly programming NeuronDevices available on AWS Trn/Inf instances. You can use NKI to develop, optimize and run new operators directly on NeuronCores while making full use of available compute and memory resources. NKI empowers ML developers to self-serve and invent new ways to use the NeuronCore hardware, starting NeuronCores v2 (Trainium1) and beyond.

  
## [Torch Script](https://docs.pytorch.org/docs/main/jit.html) ##

* [딥러닝 모델 배포하기 #01 - MLOps PipeLine과 연산 최적화 / 모델 경량화](https://happy-jihye.github.io/dl/torch-1/)
* [딥러닝 모델 배포하기 #02 - TorchScript & Pytorch JIT](https://happy-jihye.github.io/dl/torch-2/)
* [Introduction to TorchScript](https://docs.pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
* [Loading a TorchScript Model in C++](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)
* [TorchScript and PyTorch JIT | Deep Dive](https://www.youtube.com/watch?v=2awmrMRf0dA)


## [Trainium](https://aws.amazon.com/ko/ai/machine-learning/trainium/) ##

* [Train Llama2 with AWS Trainium on Amazon EKS](https://aws.amazon.com/ko/blogs/containers/train-llama2-with-aws-trainium-on-amazon-eks/)
  - https://github.com/awslabs/data-on-eks/tree/trn1-karp



## References ##

* [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#installation)
* [CPU, GPU, and NPU](https://levysoft.medium.com/cpu-gpu-and-npu-understanding-key-differences-and-their-roles-in-artificial-intelligence-2913a24d0747)
* [LLMs](https://wikidocs.net/book/13922)

----

* [Introduction to KubernetesExecutor and KubernetesPodOperator](https://medium.com/uncanny-recursions/introduction-to-kubernetesexecutor-and-kubernetespodoperator-ae9bb809e3b3)
* https://developer.arm.com/documentation/102374/0102/Data-processing---arithmetic-and-logic-operations
* [Dive into Deep Learning Compiler](https://tvm.d2l.ai/index.html#)

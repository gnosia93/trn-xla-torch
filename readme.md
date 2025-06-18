
  

 
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

#### Neuron Runtime ####
* https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/neuron-runtime/nrt-api-guide.html#nrt-api-guide


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
* [LLMs](https://wikidocs.net/book/13922)
* [Introduction to KubernetesExecutor and KubernetesPodOperator](https://medium.com/uncanny-recursions/introduction-to-kubernetesexecutor-and-kubernetespodoperator-ae9bb809e3b3)

---

* https://developer.arm.com/documentation/102374/0102/Data-processing---arithmetic-and-logic-operations
* [OpenMP](https://junstar92.tistory.com/234#:~:text=OpenMP%EB%8A%94%20Pthreads%EC%99%80%20%EA%B0%99%EC%9D%80%20API%EB%A5%BC%20%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC%20%EA%B3%A0%EC%84%B1%EB%8A%A5%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%EC%9D%84,level%EC%97%90%EC%84%9C%20%EA%B0%9C%EB%B0%9C%ED%95%A0%20%EC%88%98%20%EC%9E%88%EB%8F%84%EB%A1%9D%20OpenMP%20%EC%8A%A4%ED%8E%99%EC%9D%84%20%EC%A0%95%EC%9D%98%ED%95%98%EC%98%80%EC%8A%B5%EB%8B%88%EB%8B%A4.)

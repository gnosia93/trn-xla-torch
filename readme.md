* [TorchScript and PyTorch JIT | Deep Dive](https://www.youtube.com/watch?v=2awmrMRf0dA)

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


## AWS Accelerators ##

* [Setting up AWS Trainium for Hugging Face Transformers](https://www.philschmid.de/setup-aws-trainium)
* [Fine-tune BERT for Text Classification on AWS Trainium](https://huggingface.co/docs/optimum-neuron/tutorials/fine_tune_bert)


### 🤗 Optimum Neuron ###
* [Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index)



## HW ##

* [CPU, GPU, and NPU](https://levysoft.medium.com/cpu-gpu-and-npu-understanding-key-differences-and-their-roles-in-artificial-intelligence-2913a24d0747)



## ref ##

* [파이썬 코딩도장](https://dojang.io/course/view.php?id=7)

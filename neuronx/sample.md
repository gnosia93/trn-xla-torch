#### 패키지 설치 ####
```
pip install -U "numpy" "protobuf<4" "transformers==4.27.3" datasets==2.4.0 scikit-learn==1.2.2 evaluate==v0.4.0
mkdir ~/vt
cd ~/vt
git clone https://github.com/huggingface/transformers.git --branch v4.27.3
cd transformers/examples/pytorch/image-classification
#wget https://ud-workshop.s3.amazonaws.com/compiler_cache.tar
#tar -xvf compiler_cache.tar
```

#### 버전 체크 ####
```
(aws_neuronx_venv_pytorch_2_6) [ec2-user@ip-172-31-76-174 ~]$ pip list | grep numpy
numpy                     1.26.4
```

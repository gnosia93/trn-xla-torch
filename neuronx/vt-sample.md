### 트랜스포머 설치 ###
```
source aws_neuron_venv_pytorch/bin/activate 
pip install -U "numpy" "protobuf<4" "transformers==4.27.3" datasets==2.4.0 scikit-learn==1.2.2 evaluate==v0.4.0
mkdir ~/vt
cd ~/vt
git clone https://github.com/huggingface/transformers.git --branch v4.27.3
cd transformers/examples/pytorch/image-classification
#wget https://ud-workshop.s3.amazonaws.com/compiler_cache.tar
#tar -xvf compiler_cache.tar

pip list | grep numpy
numpy                     1.26.4
```

### 트레이닝 ###

#### 싱글 노드 트레이닝 ####
```
# 컴파일
XLA_USE_BF16=1 neuron_parallel_compile \
    python3 run_image_classification.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --dataset_name beans \
    --do_train \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 1337 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --output_dir vit-image-classification
```
![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/neuron-cc-1.png)


#### 분산 트레이닝 ####
```
# 컴파일
XLA_USE_BF16=1 NEURON_CC_FLAGS="--cache_dir=./compiler_cache" neuron_parallel_compile \
    torchrun --nproc_per_node=2 run_image_classification.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --dataset_name beans \
    --do_train \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 1337 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --output_dir vit-image-classification

# 훈련 실행
XLA_USE_BF16=1 NEURON_CC_FLAGS="--cache_dir=./compiler_cache" \
    torchrun --nproc_per_node=2 run_image_classification.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --dataset_name beans \
    --do_train \
    --do_eval \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 1337 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --output_dir vit-image-classification
```
```
source aws_neuron_venv_pytorch/bin/activate 
neuron-top
```
![](https://github.com/gnosia93/trn-xla-torch/blob/main/neuronx/images/neuron-top-1.png)

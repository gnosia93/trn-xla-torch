#### 패키지 설치 ####
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

#### 컴파일 ####
```
XLA_USE_BF16=1 neuron_parallel_compile NEURON_CC_FLAGS=\"--cache_dir=./compiler_cache --model-type transformer\" \
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

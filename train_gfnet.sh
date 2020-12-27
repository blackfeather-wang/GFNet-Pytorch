CUDA_VISIBLE_DEVICES=0,1 python train.py --data_url /home/data/ImageNet/ --train_stage 1 --model_arch mobilenetv3_large_100 --patch_size 128 --T 3 --print_freq 1 --model_prime_path ./mobilenet-v3/Initialization_model_prime_mobilenetv3_large_100_patch_size_128.pth  --model_path ./mobilenet-v3/Initialization_model_mobilenetv3_large_100.pth



 CUDA_VISIBLE_DEVICES=0,1 python train.py --data_url /home/data/ImageNet/ --train_stage 2 --model_arch mobilenetv3_large_100 --patch_size 96 --T 3 --print_freq 1



CUDA_VISIBLE_DEVICES=0,1 python train.py --data_url /home/data/ImageNet/ --train_stage 2 --model_arch mobilenetv3_large_100 --patch_size 96 --T 3 --print_freq 1 --checkpoint_path ./mobilenet-v3/mobilenetv3_large_100_patch_size_96_T_3_stage1.pth.tar



CUDA_VISIBLE_DEVICES=0 python train.py --data_url /home/data/ImageNet/ --train_stage 2 --model_arch mobilenetv3_large_100 --patch_size 96 --T 3 --print_freq 10 --checkpoint_path ./mobilenet-v3/mobilenetv3_large_100_patch_size_96_T_3_stage1.pth.tar

CUDA_VISIBLE_DEVICES=0 python train.py --data_url /home/data/ImageNet/ --train_stage 3 --model_arch mobilenetv3_large_100 --patch_size 128 --T 3 --print_freq 10 --checkpoint_path ./mobilenet-v3/mobilenetv3_large_100_patch_size_128_T_3_stage2.pth.tar







CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_url /home/data/ImageNet/ --train_stage 1 --model_arch resnet50 --patch_size 96 --T 5 --print_freq 10 --model_prime_path ../resnet/Initialization_model_prime_resnet50_patch_size_96\:128.pth  --model_path ../resnet/Initialization_model_resnet50.pth


CUDA_VISIBLE_DEVICES=0,1 python train.py --data_url /home/data/ImageNet/ --train_stage 2 --model_arch resnet50 --patch_size 96 --T 5 --print_freq 10 --checkpoint_path ../resnet/resnet50_patch_size_96_T_5_stage1.pth.tar


CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_url /home/data/ImageNet/ --train_stage 3 --model_arch resnet50 --patch_size 96 --T 5 --print_freq 10 --checkpoint_path ../resnet/resnet50_patch_size_96_T_5_stage2.pth.tar

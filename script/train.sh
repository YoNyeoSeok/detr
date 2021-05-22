git checkout bottomup_div3

# cluster 17 screen res50 AdamW 0
# train_bottomup
# optimizer adamw
# original image
# NAME="bottomup_swig_div3"
# OPTIM="AdamW"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --use_env \
#     main.py --batch_size 32 --dataset_file swig --image_dir images_512 --epochs 50 --backbone resnet50 \
#     --optimizer ${OPTIM} --lr 1e-4 --lr_backbone 1e-5 \
#     --dist_url file://${PWD}/dist/${NAME} \
#     --output_dir log/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log

# cluster 17 screen vgg16 AdamW 1
# NAME="bottomup_swig_div3_vgg16"
# OPTIM="AdamW"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}
# CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=2 --use_env \
#     main.py --batch_size 16 --dataset_file swig --image_dir images_512 --epochs 50 --backbone vgg16 \
#     --optimizer ${OPTIM} --lr 1e-4 --lr_backbone 1e-5 \
#     --dist_url file://${PWD}/dist/${NAME} \
#     --output_dir log/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log

# cluster 3 swig resized 256 screen AdamW
# NAME="bottomup_swig_div3_resized256"
# OPTIM="AdamW"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}
# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --use_env \
#     main.py --batch_size 32 --dataset_file swig --image_dir images_512 --epochs 50 --backbone resnet50 \
#     --optimizer ${OPTIM} --lr 1e-4 --lr_backbone 1e-5 \
#     --dist_url file://${PWD}/dist/${NAME} \
#     --output_dir log/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log

# cluster 5 swig resized 256 vgg16 screen AdamW
NAME="bottomup_swig_div3_resized256_vgg16_bn"
OPTIM="AdamW"
rm -f log/${NAME}.log dist/${NAME}
rm -rf log/${NAME}
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --use_env \
    main.py --batch_size 16 --dataset_file swig --image_dir images_512 --epochs 50 --backbone vgg16_bn \
    --optimizer ${OPTIM} --lr 1e-4 --lr_backbone 1e-5 \
    --dist_url file://${PWD}/dist/${NAME} \
    --output_dir log/${NAME} |\
    while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
    tee -a log/${NAME}.log

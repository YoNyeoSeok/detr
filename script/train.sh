# cluster 17 screen AdamW
NAME="srtr_vgg16bn_resize512"
rm -rf --output_dir log/${NAME}
rm -f dist/${NAME} log/${NAME}.log
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --epochs 50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --dataset_file swig \
    --image_dir images_512 \
    --backbone vgg16_bn \
    --image_resize 512 \
    --batch_size 8 \
    --dist_url file://${PWD}/dist/${NAME} \
    --output_dir log/${NAME} \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done \
    | tee -a log/${NAME}.log
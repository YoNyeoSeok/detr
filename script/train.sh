# cluster 17 screen AdamW
NAME="srtr_resnet50_resize512"
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --epochs 50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --dataset_file swig \
    --image_dir images_512 \
    --backbone resnet50 \
    --image_resize 512 \
    --batch_size 16 \
    --output_dir log/${NAME} \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done \
    | tee -a log/${NAME}.log
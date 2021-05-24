git checkout new_v1role190_images512_min512
rm -rf dist
mkdir dist

# images512
IMG_FOLDER="images512"
IMG_RESIZE="big"
NAME="new_v1role190_fld${IMG_FOLDER}_resize${IMG_RESIZE}"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --use_env \
    main.py --backbone vgg16_bn --optimizer AdamW --dataset_file swig --image_resize big --image_folder images --batch_size 4 --epochs 100 --num_verb_queries 1 --num_role_queries 190 \
    --enc_layers 6 --dec_layers 6 --dropout 0.1 --hidden_dim 256 \
    --output_dir log/${NAME} \
    --dist_url file://${PWD}/dist/${NAME} |\
    while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done|\
    tee -a log/${NAME}.log
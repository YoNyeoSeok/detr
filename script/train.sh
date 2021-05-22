git checkout bottomup

# cluster 17 screen Adam
# train_bottomup
# optimizer adam
# original image
# NAME="bottomup_adam"
# OPTIM="Adam"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}

# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --use_env \
#     main.py --batch_size 64 --epochs 50 --optimizer ${OPTIM} \
#     --output_dir log/${NAME} --dist_url file://${PWD}/dist/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log


# cluster 17 screen AdamW
# train_bottomup
# optimizer adamw
# original image
# NAME="bottomup_adamw"
# OPTIM="AdamW"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}

# CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=2 --use_env \
#     main.py --batch_size 64 --epochs 50 --optimizer ${OPTIM} \
#     --output_dir log/${NAME} --dist_url file://${PWD}/dist/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log

# cluster 3 screen Adamax
# train_bottomup
# optimizer adamax
# original image
# NAME="bottomup_adamax"
# OPTIM="Adamax"
# rm -f log/${NAME}.log dist/${NAME}
# rm -rf log/${NAME}

# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --use_env \
#     main.py --batch_size 32 --epochs 50 --optimizer ${OPTIM} \
#     --output_dir log/${NAME} --dist_url file://${PWD}/dist/${NAME} |\
#     while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
#     tee -a log/${NAME}.log

# cluster 5 screen SGD
# train_bottomup
# optimizer sgd
# original image
NAME="bottomup_sgd"
OPTIM="SGD"
rm -f log/${NAME}.log dist/${NAME}
rm -rf log/${NAME}

CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node=3 --use_env \
    main.py --batch_size 32 --epochs 50 --optimizer ${OPTIM} \
    --output_dir log/${NAME} --dist_url file://${PWD}/dist/${NAME} |\
    while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
    tee -a log/${NAME}.log

# cluster 141 screen RMSprop
# train_bottomup
# optimizer rmsprop
# original image
NAME="bottomup_rmsprop"
OPTIM="RMSprop"
rm -f log/${NAME}.log dist/${NAME}
rm -rf log/${NAME}

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --use_env \
    main.py --batch_size 64 --epochs 50 --optimizer ${OPTIM} \
    --output_dir log/${NAME} --dist_url file://${PWD}/dist/${NAME} |\
    while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done |\
    tee -a log/${NAME}.log


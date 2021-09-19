[ ! -d "log" ] && mkdir log

PART="2080ti"
NAME="gstr_pretrained"

echo ${NAME}
rm -rf log/${NAME}

python run_with_submitit.py --ngpus 4 --nodes 1 --job_dir log/${NAME} --partition ${PART} --resume detr-r50-e632da11.pth
export PATH=/usr/local/cuda/bin:$PATH
pip install -r requirements.txt
pip install deepspeed==0.9.1
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`

# ------need to change------
# DATA_PATH=/path/to/data/
IMG_DIR=/path/to/img/cocofolder/
TEXT_DIR=/path/to/text/llava.json
MODEL_DIR=/path/to/checkpoints/

# ------config------
DEEPSPEED_CONFIG=./finetune/ft_viscpm_chat/config/deepspeed/viscpm_chat_ft.json
LLM_PATH=./config/cpm-bee-10b.json


MODEL_NAME=ft_viscpm_chat


OPTS=""
# OPTS+=" --data_path ${DATA_PATH}"
OPTS+=" --img_path ${IMG_DIR}"
OPTS+=" --text_path ${TEXT_DIR}"
OPTS+=" --llm_path ${LLM_PATH}"
OPTS+=" --exp_name ${MODEL_NAME}"
OPTS+=" --model_dir ${MODEL_DIR}"
OPTS+=" --query_num 64"
OPTS+=" --max_len 512"
OPTS+=" --batch_size 1"
OPTS+=" --save_step 500"
OPTS+=" --epochs 5"
OPTS+=" --deepspeed_config ${DEEPSPEED_CONFIG}"
OPTS+=" --sft"
OPTS+=" --tune_llm"
OPTS+=" --tune_vision"


OPTS+=" $@"
CMD="deepspeed ./finetune/ft_viscpm_chat/train_viscpm_chat.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD

export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/
BATCHSIZE=1

# Set maxshot w.r.t. context length
if [[ "${LLM}" == "gpt2-xl" ]] || [[ "{$LLM}" == "gpt2-large" ]]; then
# max context length = 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr trec) # maxshot = 8
array4=(rte) # maxshot = 4
array5=(agnews cb) # maxshot = 2
array6=(dbpedia) # maxshot = 1
else
# max context length = 2048
array1=(sst2 mpqa)
array2=(subj cr mr trec)
array3=(rte)
array4=(agnews cb)
array5=(none)
array6=(dbpedia)
fi

# for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do

DATASET=sst2

if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
NSHOT=32
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
NSHOT=16
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
NSHOT=8
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
NSHOT=4
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
NSHOT=2
else
NSHOT=1
fi

for SEED in 1 2 3 4 5; do

python3 icl.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${NSHOT} \
    --seed ${SEED} \
    --output_dir ./output

done
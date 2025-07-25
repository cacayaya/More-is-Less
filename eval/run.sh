device=${1:-"0,1,2,3"}
model=${2}
echo export CUDA_VISIBLE_DEVICES=${device}
export CUDA_VISIBLE_DEVICES=${device}

# example
python -m eval \
--eval_examples 100 \
--n_shots 0 \
--save_dir results/$(basename ${model}) \
--model_name_or_path ${model} \
--eval \
--metric gpt4o \
--eval_batch_size 1 



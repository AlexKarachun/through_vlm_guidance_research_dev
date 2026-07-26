#!/usr/bin/env bash

PROMPTS_FILE="datasets/whoops50.txt"
PROMPTS_FILE_NAME="$(basename "$PROMPTS_FILE" .txt)"

# HYDRA_FULL_ERROR=1 python run.py\
#     pipeline=sd15\
#     generation=multi_prompt\
#     generation.prompts_file=datasets/whoops50.txt\
#     pipeline.params.n_inference_steps=50\
#     generation.output_folder='sd15-whoops50'



N_STEPS=50

# 18
# for guide_step in 0 20 40; do
#     for lr in 1e-2 1e-3; do
#         for cfg_scale in 1 3 7; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 12
# for guide_step in 0 20 40; do
#     for lr in 1e-2 1e-3; do
#         for cfg_scale in 5 9; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done



# 24
# for guide_step in 0 20 40; do
#     for lr in 4e-3 4e-4; do
#         for cfg_scale in 3 5 7 9; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 12
# for guide_step in 45; do
#     for lr in 1e-2 4e-3 1e-3 4e-4; do
#         for cfg_scale in 5 7 9; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# # 12
# for guide_step in 0 10; do
#     for lr in 2e-4 1e-4; do
#         for cfg_scale in 5 7 9; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# # 12
# for guide_step in 0 10; do
#     for lr in 1e-3 4e-4 2e-4; do
#         for cfg_scale in 11 13; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# # 8
# for cfg_scale in 1 3 5 7 9 11 13 15 17 19 21; do
#     HYDRA_FULL_ERROR=1 python run.py\
#         device=cuda:0\
#         generation.output_folder="sd15-${PROMPTS_FILE_NAME}-cfg${cfg_scale}"\
#         pipeline=sd15\
#         pipeline.params.cfg_scale="${cfg_scale}"\
#         generation=multi_prompt\
#         generation.prompts_file="${PROMPTS_FILE}"\
#         pipeline.params.n_inference_steps=50\
# done

# # 3
# for guide_step in 0; do
#     for lr in 4e-4 2e-4 1e-4; do
#         for cfg_scale in 1; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:1\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 1
# for guide_step in 10; do
#     for lr in 4e-4; do
#         for cfg_scale in 3; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 1
# for guide_step in 10; do
#     for lr in 4e-4; do
#         for cfg_scale in 9; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 3
# for guide_step in 20; do
#     for lr in 1e-3 4e-4 2e-4; do
#         for cfg_scale in 13; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 1
# for guide_step in 10; do
#     for lr in 4e-3; do
#         for cfg_scale in 13; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# 24
# for guide_step in 10 20 30; do
#     for lr in 4e-3 1e-3 4e-4 2e-4; do
#         for cfg_scale in 15 17; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done






# for guide_step in 40; do
#     for lr in 1e-5 4e-5 1e-4 4e-4 1e-3 4e-3 1e-2; do
#         for cfg_scale in 13 15 17 19 21; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done





# for guide_step in 30 40; do
#     for lr in 1e-4 4e-4 1e-3 4e-3 1e-2; do
#         for cfg_scale in 13; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done



# for guide_step in 0 10 20 30; do
#     for lr in 4e-5 1e-4; do
#         for cfg_scale in 13 15; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done





# for guide_step in 20; do
#     for lr in 2e-4 4e-4 1e-3 2e-3 4e-3; do
#         for cfg_scale in 11; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# for guide_step in 10 20 30; do
#     for lr in 1e-2; do
#         for cfg_scale in 15; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:0\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done




# for guide_step in 10; do
#     for lr in 4e-4; do
#         for cfg_scale in 17; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:1\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# for guide_step in 10; do
#     for lr in 2e-4 1e-4; do
#         for cfg_scale in 15 17; do
#             output_folder="guided_sd15-${PROMPTS_FILE_NAME}-guide_steps${guide_step}${guide_step}${guide_step}${guide_step}-lr${lr}-cfg${cfg_scale}"

#             HYDRA_FULL_ERROR=1 python run.py\
#                 device=cuda:1\
#                 pipeline=guided_sd15\
#                 generation=multi_prompt\
#                 generation.prompts_file="${PROMPTS_FILE}"\
#                 generation.output_folder="${output_folder}"\
#                 pipeline.params.n_inference_steps="${N_STEPS}"\
#                 pipeline.params.cfg_scale="${cfg_scale}"\
#                 "pipeline.guidance.steps_to_guide=[${guide_step},${guide_step},${guide_step},${guide_step}]"\
#                 pipeline.optimizer.lr="${lr}"
#         done
#     done
# done


# # 8
# for cfg_scale in 11; do
#     HYDRA_FULL_ERROR=1 python run.py\
#         device=cuda:0\
#         generation.output_folder="sd15-${PROMPTS_FILE_NAME}-cfg${cfg_scale}"\
#         pipeline=sd15\
#         pipeline.params.cfg_scale="${cfg_scale}"\
#         generation=multi_prompt\
#         generation.prompts_file="${PROMPTS_FILE}"\
#         pipeline.params.n_inference_steps=50
# done


HYDRA_FULL_ERROR=1 python run.py\
    pipeline=qwen3_judge\
    pipeline.model.model_id=Qwen/Qwen3-VL-32B-Instruct\
    generation=multi_prompt


# HYDRA_FULL_ERROR=1 python run.py\
#     pipeline=qwen3_judge\
#     generation=multi_prompt



# база запуска guided multi prompt теста
# HYDRA_FULL_ERROR=1 python run.py\
#     pipeline=guided_sd15\
#     generation=multi_prompt\
#     generation.prompts_file=datasets/whoops50.txt\
#     generation.output_folder='guided_sd15-whoops50-guide_steps0000-lr1e-2-cfg7.5'
#     pipeline.params.n_inference_steps=50\
#     pipeline.params.cfg_scale=7.5\
#     'pipeline.guidance.steps_to_guide=[0, 0, 0, 0]'\
#     pipeline.optimizer.lr=1e-2\
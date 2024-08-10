# Run instructions in a foor loop :

# declare -a prompt_method=("pre"  "mid_no_step" "post")
# declare -a model=("gpt-3.5-turbo" )
# declare -a data=("doable_task" "undoable_task")

# for m in "${model[@]}"; do
#     for p in "${prompt_method[@]}"; do
#         for d in "${data[@]}"; do
#             echo "Running: python -m src.generate --prompt $p --model $m --data $d --fraction 1"
#             python -m src.generate --prompt $p --model $m --data $d --fraction 1
#             sleep 30
#         done
#     done
# done

# Run instructions for a single case:

# python -m src.generate --prompt pre --model gpt-3.5-turbo --data undoable_task_2k --fraction=1




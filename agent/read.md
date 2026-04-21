conda run -n interndata python agent/task_generator.py \
  --ref workflows/simbox/core/configs/tasks/basic/lift2/arrange_the_tableware/arrange_the_tableware_part0.yaml \
  --instruction "Replace fork with knife" \
  --provider openai \
  --fast

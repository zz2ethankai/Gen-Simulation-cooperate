
# 先杀掉所有后台 codex 噪音进程（IDE 的 codex daemon，不影响 agent）
# 然后跑 agent
python -m agent plan --prompt "把白色杯子放到托盘里" 2>&1 &
PID=$!
sleep 3

# 看 agent 自己写了什么
RUN_DIR=$(ls -dt output/agent_runs/*/ | head -1)
echo "Run: $RUN_DIR"
ls -la "$RUN_DIR/decisions/"
cat "$RUN_DIR/decisions/resolution.prompt_size.txt" 2>/dev/null
cat "$RUN_DIR/decisions/resolution.api_error.txt" 2>/dev/null

# 等 2 分钟看结果
sleep 120
ls -la "$RUN_DIR/decisions/"
cat "$RUN_DIR/decisions/resolution.response.json" 2>/dev/null | head -20

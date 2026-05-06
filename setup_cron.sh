#!/bin/bash
# 设置自动定时任务
# 使用方法: bash setup_cron.sh

PYTHON=$(which python3)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$PROJECT_DIR/logs/cron.log"

# 任务1：每个工作日 15:10 跑模拟交易
CRON_TRADE="10 15 * * 1-5 cd $PROJECT_DIR && $PYTHON main.py paper-trade --symbols 000001,600519,000858 --strategy composite >> $LOG 2>&1"

# 任务2：每月1号 08:00 自动优化策略参数
CRON_OPT="0 8 1 * * cd $PROJECT_DIR && $PYTHON main.py auto-optimize --symbols 000001,600519,000858 >> $LOG 2>&1"

EXISTING=$(crontab -l 2>/dev/null)

add_if_missing() {
    local job="$1"
    local keyword="$2"
    if echo "$EXISTING" | grep -q "$keyword"; then
        echo "已存在: $keyword，跳过"
    else
        EXISTING="$EXISTING
$job"
        echo "已添加: $job"
    fi
}

add_if_missing "$CRON_TRADE" "paper-trade"
add_if_missing "$CRON_OPT"   "auto-optimize"

echo "$EXISTING" | crontab -

echo ""
echo "当前所有定时任务:"
crontab -l
echo ""
echo "日志: $LOG"

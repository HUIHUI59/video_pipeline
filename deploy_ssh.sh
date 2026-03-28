#!/bin/bash

# ================= 1. 核心节点配置区 =================
# 请在这里填入 A6000 和 A8000 宿主机的 Windows 局域网物理 IP
NODE_A6000_IP="192.168.50.102" 
NODE_A8000_IP="192.168.50.237"
# 目标机器的 WSL 用户名
TARGET_USER="leolee"
# 你在端口转发中设置的监听端口
FORWARD_PORT="2222"

echo "[SYSTEM] 启动分布式节点底层控制权部署..."

# ================= 2. 原子化操作：生成与投递 =================
# 静默生成主控机密钥（如果已存在则自动跳过，绝不卡顿询问）
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -C "distributed-pipeline" -f ~/.ssh/id_ed25519 -N "" -q
    echo "[SYSTEM] 主控机核心 SSH 密钥对已生成。"
fi

# 开始投递密钥（注意：这是整个流程中唯一需要你手动输入目标机器密码的环节，用于初次握手）
echo "[SYSTEM] 正在向 A6000 节点投递控制凭证..."
# 关键点：必须使用 -p 参数强行指定转发端口
ssh-copy-id -p $FORWARD_PORT -i ~/.ssh/id_ed25519.pub $TARGET_USER@$NODE_A6000_IP

echo "[SYSTEM] 正在向 A8000 节点投递控制凭证..."
ssh-copy-id -p $FORWARD_PORT -i ~/.ssh/id_ed25519.pub $TARGET_USER@$NODE_A8000_IP


# ================= 3. 多重验证：免密穿透测试 =================
echo -e "\n[SYSTEM] 投递完毕，进入免密通道严格自检..."

# 遍历所有节点进行测试
for IP in $NODE_A6000_IP $NODE_A8000_IP; do
    echo "  -> 探测节点: $IP:$FORWARD_PORT"
    
    # 核心自检逻辑：BatchMode=yes 会强制禁止弹出密码输入框。
    # 如果免密配置有任何瑕疵，这条命令会直接报错失败，绝不会假死等待。
    if ssh -p $FORWARD_PORT -o BatchMode=yes -o ConnectTimeout=5 $TARGET_USER@$IP "echo 'Auth_Success_Verified'" > /dev/null 2>&1; then
        echo "     [PASS] 节点 $IP 免密控制通道已彻底打通！"
    else
        echo "     [FATAL ERROR] 节点 $IP 穿透失败！请检查端口映射或防火墙放行状态。"
        exit 1
    fi
done

echo -e "\n[SUCCESS] 集群底层通信网络已全部锁定。"
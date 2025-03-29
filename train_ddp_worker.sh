set -x

export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IPV4_ONLY=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_IB_GID_INDEX=3
ulimit -n 10240000
echo "Network settings are configured."
export WANDB_MODE=offline

hostname=$1
hostfile=$2
node_rank=$3
num_machines=$4
master_ip=$5
num_processes=$((8 * num_machines))




IP=$(hostname -I | awk '{print $1}')
echo $IP

PORT=20687
NPROC_PER_NODE=8
NNODES=$num_machines

WORK_DIR=/nfs-130/zhangguiwei/mar_1024

nslookup $hostname

cd $WORK_DIR



cd /nfs-130/zhangguiwei/mar_1024/flow_matching
/opt/miniconda3/bin/python -m pip install -e .
cd $WORK_DIR




echo "node_rank: '$node_rank', num_machines: '$num_machines', num_processes: '$num_processes', master_ip: '$master_ip'"
/opt/miniconda3/bin/python -m pip install -r requirements.txt

/opt/miniconda3/bin/torchrun   --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$node_rank --master_addr=$hostname --master_port=$PORT main_mar.py




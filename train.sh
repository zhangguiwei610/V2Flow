set -x
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_IB_GID_INDEX=3
ulimit -n 10240000
echo "Network settings are configured."
export WANDB_MODE=offline
hostname=$1
num_machines=$3
num_processes=$((8 * num_machines))
IP=$(hostname -I | awk '{print $1}')
echo $IP

PORT=20687
NPROC_PER_NODE=8
NNODES=$num_machines
WORK_DIR=/nfs-130/zhangguiwei/mar_1024

nslookup $hostname
cd $WORK_DIR
/opt/miniconda3/bin/python -m pip install -r requirements.txt



log_dir=$WORK_DIR/logs_ark

mkdir $log_dir
chmod -R 777 $log_dir
chmod -R 777 /nfs-130/zhangguiwei/mar_1024

cd $WORK_DIR/flow_matching
/opt/miniconda3/bin/python -m pip install -e .
cd $WORK_DIR

for node_rank in $(seq 1 $((NNODES - 1)))
do
    replacement="-$node_rank."
    new_hostname=$(echo "$hostname" | sed "s/-0\./$replacement/g")
    echo "target hostname: $new_hostname"
    ssh $new_hostname "cd $WORK_DIR; nohup /bin/bash $WORK_DIR/train_ddp_worker.sh '$1' '$2' '$node_rank' '$num_machines' '$IP' > '$log_dir/log_node$node_rank.txt' 2>&1 &"
done


/opt/miniconda3/bin/torchrun  --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=0 --master_addr=$hostname --master_port=$PORT main_mar.py




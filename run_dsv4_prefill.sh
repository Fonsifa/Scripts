nic_name="enp189s0f0" # change to your own nic name
local_ip=10.176.30.106 # change to your own ip

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_VERSION=0.28.0 
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.0-beta.2/python/site-packages/mooncake:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

vllm serve /opt/models/deepseek-v4-flash-w8a8-mtp  \
    --host 0.0.0.0 \
    --port 9211 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --trust-remote-code \
    --block-size 32 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": false}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 1
                }
        }
    }' > prefill.log

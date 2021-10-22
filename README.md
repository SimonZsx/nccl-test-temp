

## TEST 1  - test1-vpipe.py  basic bandwidth test to test nccl and gloo 


export NCCL_DEBUG=INFO


### To test NCCL bandwidth

python3 test1-vpipe.py --master_addr localhost --rank 0 --intra_server_broadcast --backend nccl  --send

python3 test1-vpipe.py --master_addr localhost --rank 1 --intra_server_broadcast --backend nccl 

tested PCIe time:  Complete time:  11.494902610778809

see detailed log in send.log and recv.log

### To test gloo bandwidth 

python3 test1-vpipe.py --master_addr localhost --rank 0 --intra_server_broadcast --backend gloo  --send

python3 test1-vpipe.py --master_addr localhost --rank 1 --intra_server_broadcast --backend gloo 

tested PCIe time: Complete time:  16.287189483642578



## TEST 2 - Run VPipe. Test with NSight system and check whether it's due to the nvidia-docker bootstrap configuration 

previous docker run command 

nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host zsxhku/cpm-nsight:v0

test this docker run command (diff: remove the --ipc=host args, reason: NCCL may leverage IPC namespace to check the NVLink status, and ModelArts Host may ban the access of this IPC namespace)

nvidia-docker run -it -v $PWD:/workspace --net=host zsxhku/cpm-nsight:v0  

Still, set NCCL information to debug level before run any test:

export NCCL_DEBUG=INFO

### Launch with Nsight Systems for GPU profiling

Nsight Systems usage: 

sudo /home/deepspeed/nsight-systems-2021.4.1/bin/nsys profile --trace=cuda,nvtx -d 120 -y [delay time (in seconds)] --sample=none --cpuctxsw=none -o my_test [my application]


Set the Nsight Systems -y [delay time] argument as a suitable delay to consider the bootup time before VPipe's computation actually runs, and then start collecting the runtime information. 

For example, -y 300 means delay 5 minutes/300 seconds before starting of trace collection. However, delay 5 minitues is only suitable for bare metal host where dependencies are already installed. 

Make sure super user (sudo) is used to avoid any wirte permission protection. 

Collect the my_test.qdrem file, which is a trace log file that can be interpreted by Nsights CLI.

Example:

sudo /home/deepspeed/nsight-systems-2021.4.1/bin/nsys profile --trace=cuda,nvtx -d 120 -y [delay time (in seconds)] --sample=none --cpuctxsw=none -o my_test launch_torch16.py --module=cpm2_4 --partition=/home/ma-user/modelarts/user-job-dir/cpm/cpm/cpm2_4/gpipe.json --sync_mode=asp --distributed_backend=nccl --lr=0.000600 --lr_policy=polynomial --weight_decay=0.000000 --epochs=1 --num_ranks_in_server=8 --config_path=/home/ma-user/modelarts/user-job-dir/cpm/cpm/cpm2_4/mp_conf.json --batch_size=12 --nproc_per_node=8 --training_script=/home/ma-user/modelarts/user-job-dir/cpm/cpm/main_with_runtime.py --print_freq=10 --data_dir=/home/ma-user/modelarts/inputs/data_dir_0/ --checkpoint_dir=/home/ma-user/modelarts/outputs/checkpoint_dir_0/



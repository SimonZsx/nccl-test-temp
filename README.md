

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




## TEST 2 - check whether it's due to the nvidia-docker bootstrap configuration 

previous docker run command 

nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host dmye/cpm:v0

test this docker run command (diff: remove the --ipc=host args, reason: NCCL may leverage IPC namespace to check the NVLink status, and ModelArts Host may ban the access of this IPC namespace)

nvidia-docker run -it -v $PWD:/workspace --net=host dmye/cpm:v0  

Still, set NCCL information to debug level before run any test:

export NCCL_DEBUG=INFO



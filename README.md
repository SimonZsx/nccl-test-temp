

## TEST 1  - test1-vpipe.py  basic bandwidth test to test nccl and gloo 


export NCCL_DEBUG=INFO


### To test NCCL bandwidth

python3 test1-vpipe.py --master_addr localhost --rank 0 --intra_server_broadcast --backend nccl  --send

python3 test1-vpipe.py --master_addr localhost --rank 1 --intra_server_broadcast --backend nccl 

tested PCIe time:  Complete time:  11.494902610778809

### To test gloo bandwidth 

python3 test1-vpipe.py --master_addr localhost --rank 0 --intra_server_broadcast --backend gloo  --send

python3 test1-vpipe.py --master_addr localhost --rank 1 --intra_server_broadcast --backend gloo 

tested PCIe time: Complete time:  16.287189483642578

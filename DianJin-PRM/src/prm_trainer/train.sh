export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

deepspeed \
    --master_port=29501 \
    prm_train.py
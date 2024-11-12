Repo where I'll be creating my own LLM from scratch. The goal is to deepen my understanding of model-architecture design, training methodologies, and core deep-learning principles.

Run first
`export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`

- Make sure to use PyTorch 2.0 or later!

To debug on a Mac, use: `./run_pre_training_e2e_on_mac.sh`
To run on a machine with an NVIDIA GPU(s), use: `./run_pre_training_e2e.sh`

Push to server with NVIDIA GPUs (ignoring contents from `temp_data/` dir):
```
rsync -avz --delete --progress --exclude 'temp_data/*' $PWD username@server_ip_address:/home/ubuntu/
```

More to come...

git clone https://github.com/thanhtlx/multitask_codet5
cd multitask_codet5
mkdir -p tmp/sota-output
unzip datasets
pip install evaluate sacrebleu
pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* tensorboardX
python run.py --from_pretrained Salesforce/codet5-base --dataset cmg --model_type task_prefix --label_type gt --alpha 0.5 --batch_size 16 --max_input_length 500 --output_rationale 
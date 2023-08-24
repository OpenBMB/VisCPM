# Fine-tuning the VisCPM-Chat Model
> To meet the needs in specific scenarios, we provide fine-tuning code for the VisCPM-Chat model. Users can fine-tune it on their private data. The fine-tuning code is available in the `ft_viscpm_chat` directory. Here's a usage example:

## Environment Setup
Refer to [Installation](../README_en.md/#⚙️-install)

## Data Preparation
- This example utilizes the [LLaVA-150K](https://llava-vl.github.io/) dataset's [Chinese translation version](https://huggingface.co/datasets/openbmb/llava_zh). You'll need to download the image data separately from the [COCO dataset official website](https://cocodataset.org/#download). The scripts for downloading can be found in `ft_viscpm_chat/get_llava150k_zh.sh`.

## Start Fine-tuning
```shell
# Note: The script might contain relative paths. Ensure you run the script from the root directory of the repository. Also, pay attention to this when modifying the dataset and model checkpoint paths.
# Fetch the dataset
bash ./finetune/ft_viscpm_chat/get_llava150k_zh.sh
# Modify the downloaded dataset and model checkpoint paths, and then fine-tune the model
bash ./finetune/ft_viscpm_chat/run_viscpm_chat_ft.sh
# node: 8
# batch_size: 8 * 1
```
- Script Details
```shell
# ./ft_viscpm_chat/run_viscpm_chat_ft.sh
# The following parameters can impact the fine-tuning results and training costs. Adjust them according to your needs:
query_num       # Number of queries
max_len         # Maximum text length
batch_size      # Training batch size
save_step       # Model save frequency in terms of steps
epochs          # Number of training epochs
deepspeed_config # Path to the deepspeed configuration file. Configuration details can be found on the deepspeed official site.
tune_llm        # Flag indicating whether to fine-tune the language model
tune_vision     # Flag indicating whether to fine-tune the vision model
# For more parameters, check in initializer.py
```

## Additional Information
- The fine-tuning code uses [deepspeed](https://www.deepspeed.ai/getting-started/) version 0.9.1 for the training environment setup.
- Currently, the code has only been tested on a Linux system. If you are fine-tuning on a different system, you might need to modify parts of the code.

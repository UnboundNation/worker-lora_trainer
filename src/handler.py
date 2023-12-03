import os
import shutil
import subprocess
import sys
import time
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket
current_dir = os.path.dirname(os.path.realpath(__file__))
kohya_ss_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../kohya_ss')
sys.path.append(kohya_ss_dir)
sys.path.append("..")
os.chdir(kohya_ss_dir)

from kohya_ss.lora_gui import train_model
os.chdir(os.path.dirname(current_dir))


from rp_schema import INPUT_SCHEMA



blip_caption_weights = os.path.abspath("model_cache/model_large_caption.pth")
sdxl_base_location = os.path.abspath("model_cache/sd_xl_base_1.0.safetensors")

def handler(job):

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']

    # Download the zip file
    downloaded_input = rp_download.file(job_input['zip_url'])
    command_file_path = "logs/print_command.txt"
    open(command_file_path, mode='w').close()
    if os.path.exists('./training'):
        shutil.rmtree('./training')

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')
    logging_dir = os.path.abspath(os.path.join('./training', 'logs'))
    model_dir = os.path.abspath(os.path.join('./training', 'model'))

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}"
    os.makedirs(flat_directory, exist_ok=True)
    flat_directory = os.path.abspath(flat_directory)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    subprocess.run(f"""python3 "finetune/make_captions.py" \
            --batch_size="1" \
            --num_beams="1" \
            --top_p="0.9" \
            --max_length="75" \
            --min_length="5" \
            --beam_search \
            --caption_extension=".txt" \
            "{flat_directory}" \
            --caption_weights="{blip_caption_weights}" """, cwd="kohya_ss", shell=True, check=True)

    time.sleep(5)
    flat_directory_parent = os.path.dirname(flat_directory)



    parameters_dict = {
    "headless": {
        "label": "True"
    },
    "print_only": {
        "label": "True"
    },
    "pretrained_model_name_or_path": sdxl_base_location,
    "v2": False,
    "v_parameterization": False,
    "sdxl": True,
    "logging_dir": logging_dir,
    "train_data_dir": flat_directory_parent,
    "reg_data_dir": "", #TODO !!! needs to be adjusted
    "output_dir": model_dir,
    "max_resolution": "1024,1024",
    "learning_rate": 0.0003,
    "lr_scheduler": "constant",
    "lr_warmup": 0,
    "train_batch_size": 1,
    "epoch": 5,
    "save_every_n_epochs": 100,
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "seed": "",
    "num_cpu_threads_per_process": 2,
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "caption_extension": ".txt",
    "enable_bucket": True,
    "gradient_checkpointing": True,
    "full_fp16": False,
    "no_token_padding": False,
    "stop_text_encoder_training_pct": 0,
    "min_bucket_reso": 256,
    "max_bucket_reso": 2048,
    "xformers": "xformers",
    "save_model_as": "safetensors",
    "shuffle_caption": False,
    "save_state": False,
    "resume": "",
    "prior_loss_weight": 1.0,
    "text_encoder_lr": 0.0003,
    "unet_lr": 0.0003,
    "network_dim": 256,
    "lora_network_weights": "",
    "dim_from_weights": False,
    "color_aug": False,
    "flip_aug": False,
    "clip_skip": "1",
    "gradient_accumulation_steps": 1,
    "mem_eff_attn": False,
    "output_name": "lynn",
    "model_list": sdxl_base_location,
    "max_token_length": "75",
    "max_train_epochs": "",
    "max_train_steps": "",
    "max_data_loader_n_workers": "0",
    "network_alpha": 1,
    "training_comment": "erster test",
    "keep_tokens": "0",
    "lr_scheduler_num_cycles": "",
    "lr_scheduler_power": "",
    "persistent_data_loader_workers": False,
    "bucket_no_upscale": True,
    "random_crop": False,
    "bucket_reso_steps": 64,
    "v_pred_like_loss": 0,
    "caption_dropout_every_n_epochs": 0.0,
    "caption_dropout_rate": 0,
    "optimizer": "Adafactor",
    "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False",
    "lr_scheduler_args": "",
    "noise_offset_type": "Original",
    "noise_offset": 0,
    "adaptive_noise_scale": 0,
    "multires_noise_iterations": 0,
    "multires_noise_discount": 0,
    "LoRA_type": "Standard",
    "factor": -1,
    "use_cp": False,
    "decompose_both": False,
    "train_on_input": True,
    "conv_dim": 1,
    "conv_alpha": 1,
    "sample_every_n_steps": 0,
    "sample_every_n_epochs": 0,
    "sample_sampler": "euler_a",
    "sample_prompts": "",
    "additional_parameters": "",
    "vae_batch_size": 0,
    "min_snr_gamma": 0,
    "down_lr_weight": "",
    "mid_lr_weight": "",
    "up_lr_weight": "",
    "block_lr_zero_threshold": "",
    "block_dims": "",
    "block_alphas": "",
    "conv_block_dims": "",
    "conv_block_alphas": "",
    "weighted_captions": False,
    "unit": 1,
    "save_every_n_steps": 0,
    "save_last_n_steps": 0,
    "save_last_n_steps_state": 0,
    "use_wandb": False,
    "wandb_api_key": "",
    "scale_v_pred_loss_like_noise_pred": False,
    "scale_weight_norms": 0,
    "network_dropout": 0,
    "rank_dropout": 0,
    "module_dropout": 0,
    "sdxl_cache_text_encoder_outputs": False,
    "sdxl_no_half_vae": True,
    "full_bf16": False,
    "min_timestep": 0,
    "max_timestep": 1000,
    "vae": "",
    "debiased_estimation_loss": False,
}

    ordered_parameters = list(parameters_dict.values())
    #generate command
    train_model(*ordered_parameters)
    with open(command_file_path, 'r',) as file:
        commandtxt = file.read()

    subprocess.run(commandtxt, cwd="kohya_ss", shell=True, check=True)
    return




    #OLD STUFF
    #--reg_data_dir="/workspace/Unbound Data/Lynn/reg" \
    '''subprocess.run(f"""accelerate launch --num_cpu_threads_per_process=2 "sdxl_train_network.py" \
                --enable_bucket \
                --min_bucket_reso=256 \
                --max_bucket_reso=2048 \
                --pretrained_model_name_or_path="{sdxl_base_location}" \
                --train_data_dir="{flat_directory_parent}" \
                --resolution="1024,1024"  \
                --output_dir="{model_dir}" \
                --logging_dir="{logging_dir}" \
                --network_alpha="1" \
                --training_comment="trigger comment: TOK" \
                --save_model_as=safetensors \
                --network_module=networks.lora \
                --text_encoder_lr=0.0003 \
                --unet_lr=0.0003 \
                --network_dim=256 \
                --output_name="Lynn" \
                --lr_scheduler_num_cycles="5" \
                --no_half_vae \
                --learning_rate="0.0003" \
                --lr_scheduler="constant" \
                --train_batch_size="1" \
                --max_train_steps="3000" \
                --save_every_n_epochs="1000" \
                --mixed_precision="fp16" \
                --save_precision="fp16" \
                --caption_extension=".txt" \
                --cache_latents \
                --cache_latents_to_disk \
                --optimizer_type="Adafactor" \
                --optimizer_args scale_parameter=False relative_step=False warmup_init=False \
                --max_data_loader_n_workers="0" \
                --bucket_reso_steps=64 \
                --gradient_checkpointing \
                --xformers \
                --bucket_no_upscale \
                --noise_offset=0.0""", cwd="kohya_ss", shell=True, check=True)'''



    return

    #Unclear where to save ....

    job_s3_config = job.get('s3Config')

    uploaded_lora_url = upload_file_to_bucket(
        file_name=f"{job['id']}.safetensors",
        file_location=f"./training/model/{job['id']}.safetensors",
        bucket_creds=job_s3_config,
        bucket_name=None if job_s3_config is None else job_s3_config['bucketName'],
    )

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})

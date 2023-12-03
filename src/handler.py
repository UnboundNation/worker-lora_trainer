import os
import shutil
import subprocess
import time
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

#from kohya_ss import lora_gui
from rp_schema import INPUT_SCHEMA


blip_caption_weights = "/Users/jannisbaule/www/worker-lora_trainer/model_cache/model_large_caption.pth"
sdxl_base_location = "/Users/jannisbaule/www/worker-lora_trainer/model_cache/sd_xl_base_1.0.safetensors"
def handler(job):

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']

    # Download the zip file
    downloaded_input = rp_download.file(job_input['zip_url'])
    if os.path.exists('./training'):
        import shutil
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

    #TODO: missing:
    #--reg_data_dir="/workspace/Unbound Data/Lynn/reg" \
    subprocess.run(f"""accelerate launch --num_cpu_threads_per_process=2 "sdxl_train_network.py" \
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
                --save_every_n_epochs="1" \
                --mixed_precision="bf16" \
                --save_precision="bf16" \
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
                --noise_offset=0.0""", cwd="kohya_ss", shell=True, check=True)



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

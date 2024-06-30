# DH NOTES

## demo_4M_sampler

```python
>>> from fourm.demo_4M_sampler import Demo4MSampler, img_from_url
INFO:datasets:PyTorch version 2.3.1 available.
xFormers not available
xFormers not available
No module named 'detectron2'
Detectron2 can be used for semseg visualizations. Please install detectron2 to use this feature, or plotting will fall back to matplotlib.
No module named 'smplx'
Human pose dependencies are not installed, hence poses will not be visualized. To visualize them (optional), you can do the following:
1) Install via `pip install timm yacs smplx pyrender pyopengl==3.1.4`
   You may need to follow the pyrender install instructions: https://pyrender.readthedocs.io/en/latest/install/index.html
2) Download SMPL data from https://smpl.is.tue.mpg.de/. See https://github.com/shubham-goel/4D-Humans/ for an example.
3) Copy the required SMPL files (smpl_mean_params.npz, SMPL_to_J19.pkl, smpl/SMPL_NEUTRAL.pkl) to fourm/utils/hmr2_utils/data .
```

## mps device

NotImplementedError: The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device.
As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op

## run_generation

```bash
usage: FourM generation script [--run_name RUN_NAME] [--cond_domains COND_DOMAINS] [--target_domains TARGET_DOMAINS] [--tokens_per_target TOKENS_PER_TARGET] [--autoregression_schemes AUTOREGRESSION_SCHEMES]
                               [--decoding_steps DECODING_STEPS] [--token_decoding_schedules TOKEN_DECODING_SCHEDULES] [--temps TEMPS] [--temp_schedules TEMP_SCHEDULES] [--cfg_scales CFG_SCALES] [--cfg_schedules CFG_SCHEDULES]
                               [--cfg_grow_conditioning] [--no_cfg_grow_conditioning] [--top_p TOP_P] [--top_k TOP_K] [--sr_cond_domains SR_COND_DOMAINS] [--sr_target_domains SR_TARGET_DOMAINS]
                               [--sr_tokens_per_target SR_TOKENS_PER_TARGET] [--sr_autoregression_schemes SR_AUTOREGRESSION_SCHEMES] [--sr_decoding_steps SR_DECODING_STEPS] [--sr_token_decoding_schedules SR_TOKEN_DECODING_SCHEDULES]
                               [--sr_temps SR_TEMPS] [--sr_temp_schedules SR_TEMP_SCHEDULES] [--sr_cfg_scales SR_CFG_SCALES] [--sr_cfg_schedules SR_CFG_SCHEDULES] [--sr_cfg_grow_conditioning] [--sr_no_cfg_grow_conditioning]
                               [--sr_top_p SR_TOP_P] [--sr_top_k SR_TOP_K] [--num_samples NUM_SAMPLES] [--num_variations NUM_VARIATIONS] [--seed SEED] [--detokenizer_steps DETOKENIZER_STEPS] [--rgb_tok_id RGB_TOK_ID]
                               [--depth_tok_id DEPTH_TOK_ID] [--normal_tok_id NORMAL_TOK_ID] [--edges_tok_id EDGES_TOK_ID] [--semseg_tok_id SEMSEG_TOK_ID] [--clip_tok_id CLIP_TOK_ID] [--dinov2_tok_id DINOV2_TOK_ID]
                               [--imagebind_tok_id IMAGEBIND_TOK_ID] [--dinov2_glob_tok_id DINOV2_GLOB_TOK_ID] [--imagebind_glob_tok_id IMAGEBIND_GLOB_TOK_ID] [--sam_instance_tok_id SAM_INSTANCE_TOK_ID]
                               [--human_poses_tok_id HUMAN_POSES_TOK_ID] [--text_tok_path TEXT_TOK_PATH] [--activate_controlnet] [--no_activate_controlnet] [--controlnet_id CONTROLNET_ID]
                               [--controlnet_guidance_scale CONTROLNET_GUIDANCE_SCALE] [--controlnet_cond_scale CONTROLNET_COND_SCALE] [--model MODEL] [--sr_model MODEL] [--image_size IMAGE_SIZE] [--patch_size PATCH_SIZE]
                               [--dtype {float16,bfloat16,float32,bf16,fp16,fp32}] [--data_path DATA_PATH] [--data_name DATA_NAME] [--num_workers NUM_WORKERS] [--pin_mem] [--no_pin_mem] [--parti_prompts_t5_embs PARTI_PROMPTS_T5_EMBS]
                               [--s3_endpoint S3_ENDPOINT] [--s3_path S3_PATH] [--image_size_metrics IMAGE_SIZE_METRICS] [--name NAME] [--sr_name SR_NAME] [--output_dir OUTPUT_DIR] [--num_log_images NUM_LOG_IMAGES]
                               [--save_all_outputs] [--no_save_all_outputs] [--log_wandb] [--no_log_wandb] [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--wandb_run_name WANDB_RUN_NAME] [--wandb_mode WANDB_MODE]
                               [--show_user_warnings] [--device DEVICE] [--dist_gen] [--no_dist_gen] [--world_size WORLD_SIZE] [--local_rank LOCAL_RANK] [--dist_on_itp] [--dist_url DIST_URL]
```

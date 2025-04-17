export WANDB_API_KEY=ca2ebcfcd0138634c791073c44260f2b86764de1


#python train_full.py -c anyplace_cfgs/hang_mug/anyplace_diffusion_molmocrop.yaml
#python train_full.py -c anyplace_cfgs/close_lid/anyplace_diffusion_molmocrop.yaml 
#python train_full.py -c anyplace_cfgs/stack_blocks/anyplace_diffusion_molmocrop.yaml 
#python train_full.py -c anyplace_cfgs/push_T_hex_key/anyplace_diffusion_molmocrop.yaml 
#python train_full.py -c anyplace_cfgs/pour_into_bowl/anyplace_diffusion_molmocrop.yaml

python train_full.py -c anyplace_cfgs/5tasks/anyplace_diffusion_molmocrop.yaml 


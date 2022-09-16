#for i in {1..28}
#do
#    nohup python -u supervised_data.py --filename=/data/honerkam/pfnet_data/train_navagent/Rs0_floor0_${i}.tfrecord --scene_id=Rs --agent=goalnav_agent --num_records=268 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs --config_file=./configs/locobot_pfnet_nav.yaml --env_mode=headless --device_idx=3 --seed=90 &> nohup_datacollection${i}.out &
#done
#for i in {1..28}
#do
#    echo "${i}";
#done

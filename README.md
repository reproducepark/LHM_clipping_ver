# Clipping_LHM

Original: https://github.com/aigc3d/LHM

For installation and env settings, refer to original repo.

For low VRAM GPU, Clipping video processing on video to motion added.

HIGH processing time

```
./engine/pose_estimation/v2mseg.py --video_path ./myvideos/<your_video>.mp4 --output_path ./myoutputs
bash inference.sh LHM-500M train_data/example_imgs/<full_body_img>.jpg  myoutputs/<your_video>/smplx_params
```


----------------------------------------------

Updated : I added background merging process with ffmpeg. Auto background resizing.

```
python merge_background.py <background_img_path>.jpg <forground_video_path>.mp4 <output_path>.mp4
```

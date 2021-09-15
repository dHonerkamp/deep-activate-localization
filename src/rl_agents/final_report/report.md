#### Passive Localization

| Test Name |  for House3D env | for iGibson env|
|:---------:|:----------------:|:--------------:|
| Single apartment | rgb / rgbd / depth + gaussian/uniform + more/less particles | rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |
| Multiple apartments | rgb / rgbd / depth + gaussian/uniform + more/less particles| rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |

#### Active Localization

| Test Name | for iGibson env|
|:---------:|:--------------:|
| Single apartment (fixed pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Single apartment (random pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Multiple apartments (random pose) |belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |


|                       |     RGB    |    Depth   |    RGB-D   |
|:---------------------:|:----------:|:----------:|:----------:|
|  House3D (tracking)   |   41.187   |   41.572   |   44.502   |
```
House3D RMSE(cm) for tracking - init_particles_distr 'tracking', init_particles_std '0.3' '0.523599', num_particles '300', map_pixel_in_meters '0.02', transition_std '0.' '0.', trajlen '24', resample 'false'
```
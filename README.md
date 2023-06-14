# Mr. Sum: Large-scale Video Summarization Dataset and Benchmark

Mr. Sum is a large-scale video summarization dataset, which contains 31,892 videos selected from YouTube-8M dataset and reliable frame importance score labels aggregated from 50,000+ users per video.  

<!-- <img src="images/most_replayed.jpeg" alt="Example of Most replayed" width="300" height="200"> -->


<!-- <img src="images/AC_sparta_all_gif_resized.gif" alt="Example of Soccer game Most replayed" width="200" height="250"> -->

<!-- **In this repository,**

1. We provide meta data and most replayed labels for 31,892 videos in [dataset](dataset) folder.

2. We provide Most replayed crawler enabling expansion of our dataset.

3. We provide sample codes to apply Mr.Sum dataset on a video summarization model. -->

## Most Replayed Statistics for Summarization

### Example 1: AC Sparta Praha - Top 10 goals, season 2013/2014

<p align="center">
  <img src="images/AC_sparta_most_replayed_with_numbers.png" width="50%">
</p>

| **<span style="color:yellow">1</span>** | **<span style="color:yellow">2</span>** | **<span style="color:yellow">3</span>** | **<span style="color:yellow">4</span>** |
| --- | --- | --- | --- |
| ![gif1](images/AC_sparta_1_gif.gif) | ![gif2](images/AC_sparta_2_gif.gif) | ![gif3](images/AC_sparta_3_gif.gif) | ![gif4](images/AC_sparta_4_gif.gif) |

The four most viewed scenes in the "AC Sparta Praha" video ([Link](https://youtu.be/hqm6r8xeAew)) all show players scoring goals. 

### Example 2: Best Bicyle Kick Goals in Peru

<p align="center">
  <img src="images/mejores_most_replayed_with_numbers.png" width="50%">
</p>

| **<span style="color:yellow">1</span>** | **<span style="color:yellow">2</span>** | **<span style="color:yellow">3</span>** | **<span style="color:yellow">4</span>** |
| --- | --- | --- | --- |
| ![gif1](images/mejores_1_gif.gif) | ![gif2](images/mejores_2_gif.gif) | ![gif3](images/mejores_3_gif.gif) | ![gif4](images/mejores_4_gif.gif) |

The four most viewed scenes in the above video all show players scoring goals with amazing bicycle kicks.([Link](https://youtu.be/q89vpZ1kwpM))

### Example 3: Neo - 'The One' | The Matrix

<p align="center">
  <img src="images/the_matrix_most_replayed_with_numbers.png" width="50%">
</p>

| **<span style="color:yellow">1</span>** | **<span style="color:yellow">2</span>** | **<span style="color:yellow">3</span>** |
| --- | --- | --- |
| ![gif1](images/the_matrix_1_gif.gif) | ![gif2](images/the_matrix_2_gif.gif) | ![gif3](images/the_matrix_3_gif.gif) |

In the first most viewed scene, noted as 1 in the video, as soon as Neo meets Agent Smith, he is immediately shot by a gun. The second most viewed scene, noted as 2, plenty of Agent Smiths shoots Neo and Neo reaches out his hand to block the bullets. Lastly, in the most viewed scene 3, Neo engages in combat with Agent Smith. ([Link](https://www.youtube.com/watch?v=H-0RHqDWcJE))

### Update
- **2023.06.07**, Repository created.


----
## Getting Started

1. Download the [YouTube-8M](https://research.google.com/youtube8m/) dataset and place it under your dataset path. For example, when your dataset path is `/data/dataset/`, place your `yt8m` folder under the dataset path.

2. Download [mrsum.h5](https://drive.google.com/file/d/1LmasS9joHe2VqINO2ZXLFdAE5nPdngyO/view?usp=sharing) and [metadata.csv](https://drive.google.com/file/d/1GhUSEzPif5h2sUtHsSK9zn4qlEqeKcgY/view?usp=sharing) and place it under the `dataset` folder.

3. Create a virtual environment and then run:
    ```
    pip install -r requirements.txt
    ```

----
## Complete Mr.Sum Dataset

You need four fields on your `mrsum.h5` to prepare.

1. `features`: Video frame features from the YouTube-8M dataset.
2. `gtscore`: The Most replayed statistics normalized to a score of 0 to 1.
3. `change_points`: Shot boundary information obtained using the  [Kernel Temporal Segmentation](https://github.com/TatsuyaShirakawa/KTS) algorithm.
4. `gtsummary`: Ground truth summary obtained by solving the 0/1 knapsack algorithm on shots.

We provide three fields, `gtscore`, `change_points`, and `gtsummary`, inside `mrsum.h5`. 

After downloading the YouTube-8M dataset, you can add the `features` field using
```
python preprocess/preprocess.py --dataset_path <your_dataset_path>/yt8m
```
For example, when your dataset path is `/data/dataset/`, follow the command below.
```
python preprocess/preprocess.py --dataset_path /data/dataset/yt8m
```

Please read [DATASET.md](dataset/DATASET.md) for more details about Mr.Sum.

----
## Baseline models on Mr.Sum

We provide compatible code for three baselines models, [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM), [VASNet](https://github.com/ok1zjf/VASNet), and [SL-module](https://github.com/ChrisAllenMing/Cross_Category_Video_Highlight).

You can train baseline models on Mr.Sum from scratch using the following commands.

- PGL-SUM
  ```
  python main.py --train True --model PGL_SUM --batch_size 324 --epochs 200 --tag train_scratch
  ```

- VASNet
  ```
  python main.py --train True --model VASNet --batch_size 324 --epochs 200 --tag train_scratch
  ```

- SL-module
  ```
  python main.py --train True --model SL_module --batch_size 324 --epochs 200 --tag train_scratch
  ```

Furthermore, we provide trained checkpoints of each model for reproducibility.
- [Download PGL-SUM checkpoint](https://drive.google.com/file/d/1NQ4UT-0iOBA8-GFvpGDoAgpYV8k8p2OF/view?usp=sharing)
- [Download VASNet checkpoint](https://drive.google.com/file/d/1gyv6ZuD5zvKWUbcmlKZcN720fnw8a69d/view?usp=sharing)
- [Download SL-module checkpoint](https://drive.google.com/file/d/1oUXYWAspFhjKYcWP9aN1DPYO0EUT0eh3/view?usp=sharing)

<!-- ** We will further release more checkpoints once the paper is accepted. -->

Follow the command below to run inference on trained checkpoints.
```
python main.py --train False --model <model_type> --ckpt_path <checkpoint file path> --tag inference
```
For example, if you download the VASNet checkpoint and place it inside the `dataset` folder, you can use the command as follows.
```
python main.py --train False --model VASNet --ckpt_path dataset/vasnet_try1_152.tar.pth --tag vasnet_inference
```

## Train your summarization model on Mr.Sum

We provide a sample code for training and evaluating summarization models on Mr.Sum.

Summarization model developers can test their own model by implementing pytorch models under the `networks` folder.

We provide the [`SimpleMLP`](networks/mlp.py) summarization model as a toy example.

You can train your model on Mr.Sum dataset using the command below. Modify or add new configurations with your taste!
```
python main.py --train True --batch_size 8 --epochs 50 --tag exp1
```

----

## License of Assets
This dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/) following the YouTube-8M dataset. All the Mr.Sum dataset users must comply with the [YouTube Terms of Service](https://www.youtube.com/static?template=terms) and [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service#agreement).


This code referred to [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM), [VASNet](https://github.com/ok1zjf/VASNet), and [SL-module](https://github.com/ChrisAllenMing/Cross_Category_Video_Highlight). Every part of the code from the original repository follows the corresponding license.
Our license of the code can be found in [LICENSE](LICENSE).

----

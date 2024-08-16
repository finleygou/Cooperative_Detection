# Cooperative_Detection

环境创建

```Plain
conda create -n py38 python=3.8
conda activate py38
```

查看cuda版本：

```Bash
nvidia-smi
```

上pytorch官网

Pytorch：https://pytorch.org/get-started/locally/

```Plaintext
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

不需要GPU，只用CPU版本

安装其他包

```Bash
conda install setproctitle absl-py gym wandb scipy shapely tensorboardX imageio seaborn 
pip install pyglet
```

报错缺什么安装什么

运行：

```Plain
scripts->render->render_mpe.py
sys.path.insert(0, "D:\Courses\scientific_research\activity\coverage\code\Cooperative_Detection")
```

修改成自己的路径

右键运行

![img](https://pd4j6ac3wb.feishu.cn/space/api/box/stream/download/asynccode/?code=MzAwZWRlNTcwMWUzYzQzMGRjZTZkNDRlYThiMjg3YWVfYWEyekhvZXQ2dW0wOWJhWmxnanFtNW02YVJkSmtHNTVfVG9rZW46RDE4RGJNRWZhb1ZMZUV4bTBOVmM5T1VkbmloXzE3MjM3OTgyMTk6MTcyMzgwMTgxOV9WNA)

![img](https://pd4j6ac3wb.feishu.cn/space/api/box/stream/download/asynccode/?code=YzVhNjBhYzdhNGVhOTZjMDI3YTNmMDZjMmFhNzM1YWJfMzlNdDdVSFg4dzNqNjJCMEp2MVpWQWVwTUNwUkR4VTdfVG9rZW46RHdGaGJEeFhBb2JRdjN4WE5zbGNvTVh5bnBZXzE3MjM3OTgyMTk6MTcyMzgwMTgxOV9WNA)

![img](https://pd4j6ac3wb.feishu.cn/space/api/box/stream/download/asynccode/?code=MzgyZTMxYmI0MDliZTg2ZDFiOTM3NjlkMmE3YjM2Zjhfcmwwem05WjIxeEVsbmpkcGM0TEtWbUZQenFBRndCOFFfVG9rZW46TEw5N2J6TlJrb1dDWWN4blRpeGNGMlg0bkZnXzE3MjM3OTgyMTk6MTcyMzgwMTgxOV9WNA)

**角度标准：**

右侧x正向，0-2pi

逆加顺减原则
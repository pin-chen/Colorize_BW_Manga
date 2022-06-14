# Introduction-to-AI-Final-Project

## Final Project

#### Requirements

```
pip install -r requirements.txt
```

#### Usage

- Collect images of anime:

	Open an anime and implement screenshot
	
	```
	python screenshot.py
	```
	
	We need at least 600 images.
	
- CNN.ver
	
	Train agent with 240 images by different models:
	
	```
	cd CNN.ver
	python training.py [option]
	
		--alpha			alpha_v2 model
		
		--beta			beta_v2 model
		
		--embed_vgg16	embed_vgg16 model
	```

	Infer images in Test with this agent:
	
	```
	cd CNN.ver
	python infer.py
	```

- UNET.ver
	
	Train agent with 240 images by different models:
	
	```
	cd UNET.ver
	python training.py [option]
	
		--unet			unet_vgg16 model
		
		--best			best version
	```

	Infer images in Test with this agent:
	
	```
	cd UNET.ver
	python infer.py
	```

#### Results

Color anime image which is changed to grayscale.

| Oringinal | CNN beta_v2 | U-net LeakReLU (best version) |
| -------- | -------- | -------- |
| ![](https://i.imgur.com/SQ7UFZk.png) | ![](https://i.imgur.com/64H01HC.png) | ![](https://i.imgur.com/1ybA7Qn.png) |

Color manga.

| Oringinal | U-net LeakReLU (best version) | App on Internet |
| -------- | -------- | -------- |
|![](https://i.imgur.com/bhnKHWZ.png)    | ![](https://i.imgur.com/LxgKAxg.jpg)     |![](https://i.imgur.com/JDJQ9wo.png)    |

#### References

1. https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d

2. https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5


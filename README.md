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

#### References

reference1: https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d

reference2: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5


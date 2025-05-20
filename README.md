# Steps
1. Unarchive license_plate_training_data and adjust `data.yaml`
Specify Path and correct paths for train, val and test
```
path: /Users/medreres/Desktop/university/8_sem/diploma/license_plate_training_data
train: train/images
val: valid/images
test: test/images
```

2. Download this [video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/) and put it to `assets` folder under name `traffic_sample.mp4`



3. Run docker compose


TODO
- [ ] Improve precision of Easy OCR model by training it on specific fonts associated with license plates
- [ ] Try investigating different models for license plate detection
- [ ] Try different approaches to license plate recognition
- [ ] Do not read data from other sources, only letters and numbers for license plate directly
- [ ] ? Use multi threading for computing in parallel ?
- [ ] Write tests for license plate detection and recognition


# TODO update readme
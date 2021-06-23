### Environment

After configuring cuda environment, you can create a conda environment to run example codes.

```bash
# create and enter into a conda environment
conda create -n orienmask python=3.7
conda activate orienmask

# install packages
pip install Cython
pip install -r requirements

# compile nms extension
python setup.py develop
```

### Dataset

1. Download 2017 COCO dataset from the official [website](https://cocodataset.org/#download).
    ```
    Images
        2017 Train images [118K/18GB]
        2017 Val images [5K/1GB]
        2017 Test images [41K/6GB] (optional)
    Annotations
        2017 Train/Val annotations [241MB]
        2017 Testing Image info [1MB] (optional)
    ```

2. Unzip and organize the dataset as follows:
    ```
    ├── /path/to/coco
    │   ├── annotations
    │   │   ├──instances_train2017.json
    │   │   ├──instances_val2017.json
    │   │   ├──image_info_test-dev2017.json
    │   ├── train2017
    │   ├── val2017
    │   ├── test2017
    ```

3. Create a soft link at current directory:
    ```bash
    ln -s /path/to/coco coco
    ```

4. Prepare ground truth files and sample list files for training:
    ```bash
    python utils/prepare_dataset.py
    ```


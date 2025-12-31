# Contribute to Remote Sensing Processor

Everyone is welcome to contribute, and we value everybody's contribution.
Code contributions are not the only way to help the community.
Answering questions, helping others, and improving the documentation are also immensely valuable.

However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/simonreise/remote-sensing-processor/blob/main/CODE_OF_CONDUCT.md).

This guide was heavily inspired by the awesome [transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) guide to contributing.

## Ways to contribute

There are several ways you can contribute to Remote Sensing Processor:

- Report bugs and issues
- Fix and improve the documentation
- Propose new features
- Submit pull requests that fix bugs or implement new features

Our community is small, so every contribution is very important!

## Reporting a bug

Remote Sensing Processor is still in early development, so it has a lot of code that was not properly tested, and there are probably still lot of bugs here.
We appreciate every report, because it will help us make Remote Sensing Processor more stable and reliable.

Please include the following information in your issue so we can quickly resolve it:

- A clear and concise description of the bug.
- The full traceback if an exception is raised.
- Your OS type and version and Python, PyTorch and Remote Sensing Processor versions when applicable.
- A short, self-contained, code snippet that allows us to reproduce the bug.
- Attach any other additional information, like screenshots, you think may help.

## Improving the documentation

We're always looking for improvements to the documentation that make it more clear and accurate, and you can contribute to it!
You can fix typos, add more information or add more examples at function docstrings, add more information to API documentation, improve our FAQ, installation and quickstart guides.

### Adding an example

Adding examples is extremely valuable to our community.
We would really appreciate if you share any example of how you used RSP to solve a real-world task.

The best way to add an example is adding it as a Markdown file to the `docs/examples` folder, but we would be happy to see any other format - Jupyter or Colab notebooks, python code files etc.

## Proposing a new feature

If there is a new feature you'd like to see in Remote Sensing Processor, please open an issue and describe:

- What is the motivation behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community? Whatever it is, we'd love to hear about it!

- Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.

- Provide a code snippet that demonstrates the feature's usage (if possible).

You can also contribute by adding a pull request that implements this feature.

### Add a new sensor

Now Remote Sensing Processor supports only Landsat and Sentinel-2 imagery. 
We would be happy if you would add support for other popular sensors like MODIS or Sentinel-1.

Now Remote Sensing Processor relies on `SatPy` in processing remote sensing data, so new functions that process remote sensing data are supposed to rely on it too.

The best example of how a sensor processing function should work is a `landsat` function, so you can use it as an example.

The imagery processor is supposed to support the common distribution file format for that type of imagery (e.g. ".nc" for MODIS) or STAC objects (if there are sources where you can get data from that sensor in STAC format) and save the processed data in GeoTiff format + STAC JSON metadata file.

The processor is also supposed to perform such operations as cloud and nodata masking, data clipping, reprojection and matching all the bands to the same resolution.

There are several functions for common raster operations like reprojection, setting nodata, data clipping etc. in `common.common_raster`.

### Add a new ML task

Now only Semantic Segmentation and Regression tasks are supported. However, we are planning to support Change Detection, Instance Segmentation, Panoptic Segmentation, Object Detection and Unsupervised Learning tasks in the future.

Adding these tasks is a really complex challenge. But if you want to work on any of these, it would be really great!

There are already base classes for Data Module, Torch Model and SKLearn Model in `segmentation/segmentation.py`, and a lot of helper functions in `segmentation/tiles.py`

### Add a model

Remote Sensing Processor supports a lot of models from different libraries, such as HuggingFace Transformers, Segmentation-Models-Pytorch, TorchVision, Scikit-Learn and XGBoost.

However, you can always add other models! Models are added for each task separately in a correspondent `models.py` file.

### Add a metric

Remote Sensing Processor supports a lot of metrics from TorchMetrics. 
However, some of the metrics are still not supported.
Also, some metrics can be initialized with custom arguments, so if there is a specific argument combination for a metric that is widely used but is not included in the list of the supported metrics, don't hesitate to add it!

### Add a loss

Now Remote Sensing Processor supports only few loss functions from Torch and Segmentation-Models-Pytorch.
It would be great to add more loss functions, e.g. from MonAI or other libraries. 

### Add an augmentation

Now Remote Sensing Processor supports only augmentations from TorchVision. However, it might be great to add more, e.g. from Albumentations.

## Create a pull request

Follow the steps below to start contributing:

1. Fork the repository by clicking on the Fork button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:
    ```
    git clone git@github.com:<your Github handle>/remote-sensing-processor.git
    cd remote-sensing-processor
    git remote add upstream https://github.com/simonreise/remote-sensing-processor.git
    ```

3. Create a new branch to hold your development changes:
    ```
    git checkout -b a-descriptive-name-for-my-changes
    ```
   Do not work on the main branch!

4. Create and activate a Python virtual environment:

   On Linux/Mac:
    ```
    python -m venv venv
    source venv/Scripts/activate
    ```

   On Windows:
    ```
    python -m venv venv
    \venv\Scripts\activate.bat
    ```

5. Set up a development environment by running the following command in a virtual environment:
    ```
    pip install -e ".[tests]"
    ```

6. Develop the features in your branch.
   
   Check out a PR checklist below to make sure your changes are compliant with our guidelines.

   As you work on your code, you should make sure the test suite passes. Run the tests impacted by your changes like this:

    ```
    python -m pytest tests/<TEST_TO_RUN>.py
    ```

7. Check the code style and format with Ruff:
    ```
    ruff check
    ruff format --check
    ```

8. Once you're happy with your changes, add the changed files with git add and record your changes locally with git commit:
    ```
    git add modified_file.py
    git add another_modified_file.py
    git commit
    ```
   Don't forget to add clear and meaningful commit messages!

9. To keep your copy of the code up to date with the original repository, merge the changes from upstream/main your branch before you open a pull request
    ```
    git fetch upstream
    git merge upstream/main
    ```
   
10. Push your changes to your branch:
    ```
    git push -u origin a-descriptive-name-for-my-changes
    ```
    Sometimes you might need to force push with the --force flag. But usually changes could be merged without any conflicts.

11. Now you can go to your fork of the repository on GitHub and click on Pull Request to open a pull request. 
    Make sure you tick off all the boxes on our checklist. 
    When you're ready, you can send your changes to the project maintainers for review.

### Pull request checklist

- If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked (and people viewing the issue know you are working on it).
- To indicate a work in progress please prefix the title with [WIP]. These are useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.
- Make sure existing tests pass.
- Make sure Ruff Linter and Formatter checks pass.
- All public methods must have informative docstrings.
- If adding a new feature, also add tests for it.

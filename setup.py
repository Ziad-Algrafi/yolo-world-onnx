from setuptools import setup, find_packages

setup(
    name="yolo-world-onnx",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "setuptools==69.5.1",
        "wheel==0.43.0",
        "numpy",
        "opencv-python",
        "onnxruntime-gpu",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "clip-for-odlabel",
    ],
    author="Ziad-Algrafi",
    author_email="ZiadAlgrafi@gmail.com",
    description="YOLO-World-ONNX is a Python package for running inference on YOLO-WORLD Open-vocabulary-object detection model using ONNX models. It provides an easy-to-use interface for performing inference on images and videos using onnxruntime.",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ziad-Algrafi/yolo-world-onnx",
    classifiers=[
        "Development Status :: 5 - Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    setup_requires=["setuptools>=69.5.1", "wheel>=0.43.0"],
    license="MIT",
)
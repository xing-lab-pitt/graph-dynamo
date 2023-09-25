import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graph-dynamo",
    version="0.1.0",
    author="{Yan Zhang, Xiaojie Qiu, Ke Ni, Jonathan Weissman, Ivet Bahar, Jianhua Xing",
    author_email="xing1@pitt.edu",
    description="A Python library for working with graphs in Amazon DynamoDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xing-lab-pitt/graph-dynamo/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'typing-extensions>=3.10.0.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.4',
        ]
    }
)
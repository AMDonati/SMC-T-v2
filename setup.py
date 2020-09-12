import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SMC-T-v2",
    version="0.0.1",
    author="Alice Martin",
    author_email="alice.martindonatie@gmail.com",
    description="Code for the paper The Monte Carlo Transformer: a stochastic self-attention model for sequence prediction'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AMDonati/SMC-T-v2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="olympus-security",
    version="1.0.0",
    description="OLYMPUS-SECURITY: AI-native cyber offense/defense research platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="George David Tsitlauri",
    author_email="gdtsitlauri@gmail.com",
    url="https://gdtsitlauri.dev",
    project_urls={
        "Homepage": "https://gdtsitlauri.dev",
        "Source": "https://github.com/gdtsitlauri/olympus-security",
    },
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "streamlit>=1.32.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "gpu": ["torch>=2.2.0"],
        "dev": ["pytest>=8.0.0", "black>=24.0.0", "mypy>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "olympus=olympus.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

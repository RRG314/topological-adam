from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent

# Long description from README (fallback to short if not present)
readme_path = root / "README.md"
long_desc = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "Energy-Stabilized Topological Adam Optimizer for PyTorch."

# Requirements from requirements.txt (fallback to torch if not present)
req_path = root / "requirements.txt"
if req_path.exists():
    install_requires = [ln.strip() for ln in req_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
else:
    install_requires = ["torch>=1.9.0"]

setup(
    name="topological-adam",
    version="1.0.3",
    author="Steven Reid",
    author_email="sreid1118@gmail.com",
    description="Energy-Stabilized Topological Adam Optimizer for PyTorch",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/rrg314/topological-adam",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: PyTorch",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

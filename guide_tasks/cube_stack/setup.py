import os
from glob import glob

from setuptools import find_namespace_packages, setup

package_name = "cube_stack"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_namespace_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    # The scene is registered from the share dir: scene.py beside config/ (and the USD
    # comes from block_bin's share, see config/init.yaml).
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for pattern in (f"{package_name}/**/*", "launch/**/*", "config/**/*")
        for f in glob(pattern, recursive=True)
        if os.path.isfile(f)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="András Makány",
    maintainer_email="makany.andras@uni-obuda.hu",
    description="GUIDE demonstration task: stack the cubes of the block_bin scene in a random order.",
    license="GPL-3.0-only",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "solve_task = cube_stack.solve_task:main",
        ],
    },
)

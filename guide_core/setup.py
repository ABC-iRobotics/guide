import os
from glob import glob

from setuptools import find_packages, setup

package_name = "guide_core"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join("launch", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ]
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join("config", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ]
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join("dummy_scene", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="András Makány",
    maintainer_email="makany.andras@uni-obuda.hu",
    description="GUIDE framework core components.",
    license="GPL-3.0-only",
    entry_points={
        "console_scripts": [
            "GUIDE = guide_core.ros.guide_ros:ros_entry_point",
            "guide_exo_test = guide_core.guide_exo_test:main",
            "guide_smolvla_test = guide_core.run_smolvla:main",
        ],
    },
)

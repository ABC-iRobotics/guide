import os
from glob import glob

from setuptools import find_namespace_packages, setup

package_name = "guide_ex"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_namespace_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + [
        (os.path.join("share", os.path.dirname(f)), [f])
        for f in glob(os.path.join(f"{package_name}", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="András Makány",
    maintainer_email="makany.andras@uni-obuda.hu",
    description="GUIDE-EX is a framework for high level task specification and execution for robots.",
    license="GPL-3.0-only",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
)

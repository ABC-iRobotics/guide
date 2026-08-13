import os
from glob import glob

from setuptools import find_namespace_packages, setup

package_name = "block_bin"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_namespace_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join(package_name, "**", "*"), recursive=True)
        if os.path.isfile(f)
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
        # Scene USD assets (block_bin.usd etc.) — required by the task at runtime.
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for f in glob(os.path.join("assets", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ]
    + [
        (os.path.join("share", package_name, os.path.dirname(f).replace("../../", "")), [f])
        for f in glob(os.path.join("../../modules", "irob_franka_ros2", "**", "*"), recursive=True)
        if os.path.isfile(f)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="András Makány",
    maintainer_email="makany.andras@uni-obuda.hu",
    description="Example of a GUIDE framework demonstration task for pick and place operation.",
    license="GPL-3.0-only",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "solve_task = block_bin.solve_task:main",
            "eval_policy = block_bin.eval_policy:main",
            "eval_policy_servo = block_bin.eval_policy_servo:main",
        ],
    },
)

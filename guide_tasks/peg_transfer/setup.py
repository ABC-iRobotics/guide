import os
from glob import glob

from setuptools import find_namespace_packages, setup

package_name = "peg_transfer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_namespace_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    # The scene is registered from the share dir: scene.py beside config/, and the flattened
    # scene USD (assets/peg_transfer.usd) with its licence files. assets/dvrk* and surrol/ are
    # the export's sources (tools/export_scene.py) and stay out of the install.
    + [
        (os.path.join("share", package_name, os.path.dirname(f)), [f])
        for pattern in (f"{package_name}/**/*", "launch/**/*", "config/**/*", "assets/*")
        for f in glob(pattern, recursive=True)
        if os.path.isfile(f)
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="András Makány",
    maintainer_email="makany.andras@uni-obuda.hu",
    description="GUIDE demonstration task: bimanual peg transfer on the dVRK (PSM1, PSM2, ECM).",
    license="GPL-3.0-only",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "solve_task = peg_transfer.solve_task:main",
        ],
    },
)

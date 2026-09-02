import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rammp_box_opening"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "config", "containers"),
            glob("config/containers/*.yaml"),
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RAMMP",
    maintainer_email="chrisman4247@gmail.com",
    description="Box-opening primitives over the RAMMP-CuRobo planning service",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "press_demo = rammp_box_opening.tasks.press_demo:main",
            "home_arm = rammp_box_opening.tasks.home_arm:main",
            "preflight = rammp_box_opening.tasks.preflight:main",
            "owl_detector = rammp_box_opening.perception.owl_node:main",
            "joint_state_relay = rammp_box_opening.runtime.joint_state_relay:main",
        ],
    },
)

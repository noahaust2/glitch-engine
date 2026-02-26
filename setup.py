from setuptools import setup, find_packages

setup(
    name="glitch-engine",
    version="0.2.0",
    description="Procedural sampling & glitch engine for IDM audiovisual production",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "librosa>=0.10",
        "soundfile>=0.12",
        "moviepy>=2.0",
        "Pillow>=9.0",
    ],
    extras_require={
        "rubberband": ["pyrubberband>=0.3"],
    },
    entry_points={
        "console_scripts": [
            "glitch=glitch.cli:main",
        ],
    },
)

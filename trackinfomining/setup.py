from setuptools import setup, find_packages
from os import path, environ
 
from io import open
 
here = path.abspath(path.dirname(__file__))
 
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
 
setup(
    name='trackinfomining',
    version=2.0,
    description='trajectory information mining package',
    author='sunhuiling',
    author_email='sunnyl95@163.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url = "https://github.com/sunnyl95/Anli/tree/main/trackinfomining",
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.6, <4',
    install_requires=[
        'pandas==1.1.5'
    ],
)

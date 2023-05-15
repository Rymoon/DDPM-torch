from setuptools import find_packages, setup
from pathlib import Path
import re
import json
import yaml

name = "myddpm_pytorch"
version = "0.1"
ppkg = Path(Path(__file__).parent,name)

# Create and update __init__.py
p = Path(ppkg,"__init__.py")
s_ver = f"__version__='{version}'\n"
if not p.exists():
    with open(p,"x") as f:
        f.write(s_ver)
else:
    with open(p,"r") as f:
        s=  f.read()

    o = re.search("__version__[\w\W]*?\n",s)
    if o is not None:
        s =re.sub("__version__[\w\W]*?\n",s_ver,s)
    else:
        s+=f"\n{s_ver}"
    
    with open(p,"w") as f:
        f.write(s)

# Create apps/rennet.json
_d = {
    "root_Results":Path(ppkg.parent,"Results").as_posix()
    }
p= Path(ppkg,"apps","rennet.json")
if not p.exists():
    with open(p,"x") as f:
        f.write(json.dumps(_d))
else:
    pass



with open('requirements.yml') as f:
    requirements = yaml.safe_load(f)

install_requires = []
for package in requirements['dependencies']:
    if isinstance(package, str):
        install_requires.append(package)
    elif isinstance(package, dict):
        name = list(package.keys())[0]
        version = package[name]
        install_requires.append(f"{name}{version}")
        
setup(
    name=name,
    version=version,
    author='Rymoon',
    author_email='rymoon42@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
)
import os
import subprocess
import time

from setuptools import find_packages, setup
import pkg_resources

version_file = "version.py"


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    packages = [
            str(r)
            for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__),
                 filename)))
                 ]
    print(f"Installing packages: {packages}")
    return packages


if __name__ == '__main__':
    write_version_py()
    setup(
        name='clip-diffusion-art',
        version=get_version(),
        description='Fine-tune diffusion models on custom datasets and sample with text-conditioning using CLIP guidance combined with SwinIR for super resolution.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Sreevishnu Damodaran',
        url='',
        keywords = [
            'art generation',
            'text to image',
            'diffusion models',
            'guided diffusion',
            'openai clip',
            'computer vision',
            'pytorch'
        ],
        py_modules=["clip-diffusion-art"],
        include_package_data=True,
        packages=find_packages(),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.8',
        ],
        license='MIT',
        setup_requires=['cython', 'numpy'],
        install_requires=get_requirements(),
        dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
        zip_safe=False)

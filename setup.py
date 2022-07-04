from setuptools import setup, find_packages

setup(
      name='master_project',
      version='0.0.1',
      description='Adaptive multimodal localization',
      author='Mariela Castillo',
      author_email='mariela.castillo.mc2@gmail.com',
      url='https://github.com/MarielaCastillo/master_project',
      packages=find_packages(),
      install_requires=[
                'torch',
                'torchvision'
                ]
     )
from setuptools import setup, find_packages
setup(
      name = 'ladder_network',
      #package_dir={'':'ladder_network'},
      #packages=find_packages("ladder_network"),
      packages = find_packages(), # this must be the same as the name above
      version = '0.2',
      description = 'An implementation of the Ladder Network for semi-supervised learning',
      author = 'Alexandre Boyker',
      author_email = 'aboyker@hotmail.fr',
      url = 'https://github.com/aboyker/ladder_net'

      )

from distutils.core import setup
setup(
  name = 'DataWiz',         
  packages = ['DataWiz'],   
  version = '0.3',      
  #license='BSD-3-Clause',        
  description = 'DataWiz helps new data learners, hobbyist and industry practitioners write Machine Learning code faster by doing the heavy-lifting such as data cleaning, data transformation, preparation and more! ',   # Give a short description about your library
  author = 'Koye Sodipo',                   
  author_email = 'koye.sodipo@gmail.com',      
  url = 'https://github.com/ksodipo/DataWiz',   
  download_url = 'https://github.com/ksodipo/DataWiz/archive/v0.3-beta.tar.gz',    
  keywords = ['Data Engineering', 'data preparation', 'data science', 'data cleaning'],   
  install_requires=[            
          'numpy',
          'pandas',
          'sklearn',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',  
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

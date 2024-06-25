from setuptools import setup, find_packages

setup(
    name='FragenAntwortLLMGPU',
    version='0.1.9',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF',
        'tokenizers',
        'semantic-text-splitter==0.13.3',
        'langchain',
        'langchain_community',
        'torchvision',
        'torchaudio',
        'ctransformers',
    ],

    entry_points={
        'console_scripts': [
            'process_document=FragenAntwortLLMGPU.document_processor:main',
        ],
    },
    authors='Mehrdad Almasi, Demival VASQUES FILHO, and Lars Wieneke',
    author_email='Mehrdad.al.2023@gmail.com, demival.vasques@uni.lu ,lars.wieneke@gmail.com.',
    description='A package for processing documents and generating questions and answers using LLMs on GPU and CPU.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/FragenAntwortLLMGPU',  # Update with your actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

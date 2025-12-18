from setuptools import setup, find_packages

setup(
    name='FragenAntwortLLMGPU',
    version='0.1.15',
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
        # Needed for the optional Qwen (Transformers) backend
        'transformers>=4.37.0',
    ],
    entry_points={
        'console_scripts': [
            'process_document=FragenAntwortLLMGPU.document_processor:main',
        ],
    },
    author='Mehrdad Almasi, Demival Vasques, and Lars Wieneke',
    description='A package for processing documents and generating questions and answers using LLMs on GPU and CPU.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

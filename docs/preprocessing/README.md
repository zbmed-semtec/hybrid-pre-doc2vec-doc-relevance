# Hybrid approach phrase preprocessing

In this tutorial, we will cover the phrase preprocessing step for the hybrid-dictionary-ner approach. An alternative (and compatible) more general preprocessing can be found in the medline-preprocessing repository in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb).



# Prerequisites

1. Retrieve TSV files from TREC or RELISH data sets

    - Use the recommended BioC-approach to generate .tsv files from the data sets.
    
    - Remove structure words, by using the Structure_Words_removal module in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal)..

# Steps

## Step 1: Imports

First, we need to import the libraries from the code folder. To do so, change the `repository_path` variable to indicate the root path of the repository:


```python
import os
import sys

repository_path = os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance")

sys.path.append(f"{repository_path}/code/preprocessing/")
os.chdir(repository_path)

import logging
import preprocess as pp

logging.basicConfig(format='%(asctime)s %(message)s')
```

## Step 2: Load the data and preprocess

After reading the input data, the preprocessing function is executed. The code process is described later in "Code Strategy" section.


```python
input_path = "data/RELISH/RELISH_documents_20220628_ann_swr.tsv"
#input_path = "data/TREC/TREC_documents_20220628_ann_swr.tsv"

data = pp.read_data(input_path)
data = pp.preprocess_data(data)
```

## Step 3: Save the preprocess data

The output is either stored in `.tsv` or `.npy` format. The `.tsv` is smaller in disk and faster to write.


```python
output_path = "data/RELISH/RELISH_tokens.tsv"
#output_path = "data/TREC/TREC_tokens.tsv"

pp.save_output(data, output_path, npy_format=False)
#pp.save_output(data, output_path, npy_format=True)
```

# Decision notes

## Code strategy

1. The input file must be in `.tsv` format, containing three columns: "PMID", "title", "abstract". The file is recommended to be pruned of structure words following [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

2. The code loops through every row of the input `.tsv` and preprocess the title and abstract separately. 

3. The preprocess consists in:
    
    * (Optional) If the function parameter `process_abv` is set to `True` (`False` by default), an experimental abbreviation algorithm is executed to find and combine terms like "E. coli" or "S. aureus" into a single word like "e.coli". This process is not well tested and not recommended unless necessary.

    * Lowercase everything. 

    * Tokenize space-separated words. The text is split by white spaces and only alphanumeric characters and allowed punctuation is kept.

    * Removes all special character except for the hyphens `-` (this can be modified using the function parameter `allowed_punctuation`).

    * (Optional) Saves the results as a three-dimensional numpy array and saves it as a `.npy` file.

    * (Optional) Saves the results as a three column `.tsv` file containing "PMID", "title" and "abstract".

## Decisions

* Instead of using the default phrase preprocessing found in medline-preprocessing ([here](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb)), the steps produced in here are particular for the hybrid-dictionary-ner approach. The results produced are expectd to be the same, but execution time is greatly improved in this approach. The main difference is to not include the biological tokenizer `en_core_sci_lg` from the sciSpacy module, since its use is not recommended in this approach.

* For the decisions related to the actual preprocess steps followed, please refer to the main documentation in [here](https://github.com/zbmed-semtec/medline-preprocessing#cleaning-for-word-embedding).

## Notes

The time to preprocess each dataset (TREC or RELISH) using 1 core of an Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz with 16GB of RAM running Ubuntu 20.04 LTS and Python 3.8.10 is:

* RELISH (163189 publications): 2min 31s ± 538 ms on average.

* TREC (32604 publications): 26.9 s ± 203 ms on average.

# TODO

* Add to use the file directly without the need of a jupyter notebook

**REMOVE THIS LINE BEFORE FINAL VERSION COMMIT**


```python
!jupyter nbconvert docs/preprocessing/tutorial_preprocessing.ipynb --to markdown --output README.md
```

    [NbConvertApp] WARNING | pattern 'docs/preprocessing/tutorial_preprocessing.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

